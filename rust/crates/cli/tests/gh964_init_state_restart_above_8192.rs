//! gh#964 — `simulate --init-state` must accept a restart time T* above
//! 8192 = 2^13 model time units when T* is on the output grid.
//!
//! The output-axis tolerance was an absolute 1e-12, below one ulp of t above
//! 2^13, so the resume re-seat consumed the output at T* and the run was refused
//! with a message blaming the cadence. This is the issue's reproduction, scaled
//! down: a model with `from = 8000` on a 7-day emit grid, filtered to the last
//! observation at 8182 (T* = 8189, which always worked) and at 8189
//! (T* = 8196, which was refused). Shells out to the release binary, so it
//! exercises the real simulate → pfilter → state file → simulate path.

use std::path::{Path, PathBuf};
use std::process::Command;

const PARTICLES: usize = 24;

const MODEL: &str = r#"
time_unit = 'days

compartments { S, I, R }

parameters {
  N0    : count
  R0    : positive in [1.0, 10.0]
  gamma : rate
  omega : rate
  iota  : count
  rho   : probability
  k     : positive in [0.5, 1000.0]
}

let N = S + I + R

transitions {
  infection : S --> I @ R0 * gamma * S * (I + iota) / N
  recovery  : I --> R @ gamma * I
  waning    : R --> S @ omega * R
}

init {
  S = 0.5 * N0
  I = 0.001 * N0
  R = 0.499 * N0
}

observations {
  cases {
    columns   { time : time, cases : count }
    covers    = starting_on(time, 7 'days)
    emit_schedule = every 7 'days
    projected = incidence(infection)
    cases     ~ neg_binomial(mean = rho * projected, r = k)
  }
}

simulate { from = 8000 'days  to = 8260 'days  dt = 0.25 'days }
"#;

const PARAMS: &str = "N0 = 100000\nR0 = 2.0\ngamma = 0.25\nomega = 0.005\niota = 5\nrho = 0.3\nk = 50\n";

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let b = Path::new(&manifest).join("../../target/release/camdl");
    assert!(
        b.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test`",
        b.display()
    );
    b
}

fn run(bin: &Path, args: &[String], store: &Path) -> std::process::Output {
    Command::new(bin)
        .args(args)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_OUTPUT_DIR", store)
        .output()
        .expect("spawn camdl")
}

fn s(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

/// Filter the synthetic series to observations at or before `last`, save the
/// filter's final state, and restart from it. Returns the restart's output and
/// the trajectory path.
fn restart_after(bin: &Path, dir: &Path, data: &Path, last: f64) -> (std::process::Output, PathBuf) {
    let txt = std::fs::read_to_string(data).unwrap();
    let mut lines = txt.lines();
    let mut ctx = String::from(lines.next().unwrap());
    ctx.push('\n');
    for l in lines {
        let t: f64 = l.split('\t').next().unwrap().parse().unwrap();
        if t <= last {
            ctx.push_str(l);
            ctx.push('\n');
        }
    }
    let ctx_path = dir.join(format!("ctx_{last}.tsv"));
    std::fs::write(&ctx_path, ctx).unwrap();

    let model = dir.join("sirs.camdl");
    let params = dir.join("p.toml");
    let state = dir.join(format!("fs_{last}.tsv"));
    let out = run(bin, &[
        "pfilter".into(), s(&model), "--params".into(), s(&params),
        format!("--data=cases={}", s(&ctx_path)),
        "--particles".into(), PARTICLES.to_string(), "--dt".into(), "0.25".into(),
        "--seed".into(), "1".into(), "--save-final-state".into(), s(&state),
    ], &dir.join("cas_pf"));
    assert!(out.status.success(), "pfilter failed: {}", String::from_utf8_lossy(&out.stderr));

    let traj = dir.join(format!("traj_{last}.tsv"));
    let out = run(bin, &[
        "simulate".into(), s(&model), "--params".into(), s(&params),
        "--init-state".into(), s(&state),
        "--replicates".into(), PARTICLES.to_string(),
        "--dt".into(), "0.25".into(), "--seed".into(), "1".into(),
        "-o".into(), s(&traj), "--output-dir".into(), s(&dir.join("cas_sim")),
    ], &dir.join("cas_sim"));
    (out, traj)
}

fn first_time(traj: &Path) -> f64 {
    let txt = std::fs::read_to_string(traj).unwrap();
    let mut lines = txt.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#'));
    let hdr: Vec<&str> = lines.next().unwrap().split('\t').collect();
    let i = hdr.iter().position(|h| *h == "t").expect("t column");
    lines.map(|l| l.split('\t').nth(i).unwrap().parse::<f64>().unwrap()).fold(f64::INFINITY, f64::min)
}

#[test]
fn init_state_restarts_on_grid_above_8192() {
    let bin = binary();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    std::fs::write(dir.join("sirs.camdl"), MODEL).unwrap();
    std::fs::write(dir.join("p.toml"), PARAMS).unwrap();

    let obs = dir.join("obs");
    let out = run(&bin, &[
        "simulate".into(), s(&dir.join("sirs.camdl")), "--params".into(), s(&dir.join("p.toml")),
        "--seed".into(), "1".into(), "--obs-only-dir".into(), s(&obs),
        "--output-dir".into(), s(&dir.join("cas_gen")),
    ], &dir.join("cas_gen"));
    assert!(out.status.success(), "data gen failed: {}", String::from_utf8_lossy(&out.stderr));
    let data = obs.join("cases.tsv");

    // T* = 8189 (below the old threshold) and T* = 8196 (above it): both are on
    // the 7-day grid from 8000, so both must restart, at T*.
    for (last, t_star) in [(8182.0, 8189.0), (8189.0, 8196.0)] {
        let (out, traj) = restart_after(&bin, dir, &data, last);
        assert!(
            out.status.success(),
            "restart at T* = {t_star} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(first_time(&traj), t_star, "the restarted trajectory starts at T*");
    }
}
