//! gh#833, ruling 2 / proposal Testing item 9: a trajectory row states the
//! period its flow columns were accumulated over, as `t_start`/`t_stop` beside
//! `t`, and that period is the interval the value really covers — on all three
//! forward backends, through the real CLI writers.
//!
//! What "really covers" means here, checkable from the file alone: the
//! periods tile the run contiguously (`t_stop` is the row's `t`, `t_start` the
//! previous row's), the initial-condition row is the empty `[t₀, t₀)` with
//! zero flows, and the flows summed over the tiled periods reconcile with the
//! state change over the same span — `Σ flow_infection = S(t₀) − S(t_end)`
//! for a closed SIR whose `S` leaves only by infection. Under `--dates`, the
//! date twins are the same boundaries through the calendar map.

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let p = Path::new(&manifest).join("../../target/release/camdl");
    assert!(p.exists(), "release camdl binary missing: {} - run `make build-rust` or `make test`", p.display());
    p
}

/// A closed SIR with a `baseline` scenario setting every parameter, daily
/// output over `[0, 80]`; `S` leaves only through `infection`.
fn sir_basic() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../ocaml/golden/sir_basic.ir.json")
}

/// A dated fixture (`origin = 2020-02-24`, daily output over `[0, 120]`).
fn seed_timing_dated() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../sim/tests/fixtures/seed_timing_dated.ir.json")
}

fn run(camdl: &Path, args: &[&str]) -> String {
    let out = Command::new(camdl)
        .args(args)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("camdl must invoke");
    assert!(out.status.success(), "camdl {:?} failed:\n{}", args, String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// The trajectory table from `--stdout`: header cells and data rows, the
/// leading `# version` comment skipped.
fn table(stdout: &str) -> (Vec<String>, Vec<Vec<String>>) {
    let mut lines = stdout.lines().filter(|l| !l.starts_with('#'));
    let header: Vec<String> = lines.next().expect("a header").split('\t').map(String::from).collect();
    let rows: Vec<Vec<String>> = lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split('\t').map(String::from).collect())
        .collect();
    (header, rows)
}

fn col(header: &[String], name: &str) -> usize {
    header.iter().position(|c| c == name).unwrap_or_else(|| panic!("no `{name}` column in {header:?}"))
}

fn num(row: &[String], i: usize) -> f64 {
    row[i].parse::<f64>().unwrap_or_else(|e| panic!("cell {:?} at column {i}: {e}", row[i]))
}

/// The claims that hold for every backend, from the written file alone.
fn assert_periods_tile_and_reconcile(backend: &str, stdout: &str, flow_tol: f64) {
    let (header, rows) = table(stdout);
    assert_eq!(&header[..3], ["t", "t_start", "t_stop"], "[{backend}] the time columns lead the header");
    assert!(rows.len() > 2, "[{backend}] a run has more than two output rows");

    let (t, start, stop) = (col(&header, "t"), col(&header, "t_start"), col(&header, "t_stop"));
    let s = col(&header, "S");
    let flows: Vec<usize> = header.iter().enumerate()
        .filter(|(_, c)| c.starts_with("flow_")).map(|(i, _)| i).collect();
    let inf = col(&header, "flow_infection");
    assert!(!flows.is_empty(), "[{backend}] flow columns present");

    // The initial-condition row: the empty period at t₀, flows all zero.
    let first = &rows[0];
    assert_eq!(num(first, start), num(first, t), "[{backend}] first row opens at its own t");
    assert_eq!(num(first, stop), num(first, t), "[{backend}] first row closes at its own t");
    for &f in &flows {
        assert_eq!(num(first, f), 0.0, "[{backend}] no interval precedes t₀, so its flows are zero");
    }

    // Every later row closes at its own t and opens where the previous closed:
    // the periods tile the run with no gap and no overlap.
    for w in rows.windows(2) {
        let (prev, row) = (&w[0], &w[1]);
        assert_eq!(num(row, stop), num(row, t), "[{backend}] t_stop is the row's t");
        assert_eq!(num(row, start), num(prev, t), "[{backend}] t_start is the previous row's t");
        assert!(num(row, stop) > num(row, start), "[{backend}] a period has positive width");
    }

    // The flows over the tiled periods account for the state change over the
    // whole span — the identity that only holds if each row's flow really is
    // the accumulation over exactly [t_start, t_stop).
    let s0 = num(first, s);
    let s_end = num(rows.last().unwrap(), s);
    let sum_inf: f64 = rows.iter().map(|r| num(r, inf)).sum();
    assert!(
        (sum_inf - (s0 - s_end)).abs() <= flow_tol,
        "[{backend}] Σ flow_infection over the tiled periods = {sum_inf}, but S fell by {}",
        s0 - s_end
    );
}

#[test]
fn every_backend_writes_the_period_its_flows_cover() {
    let camdl = camdl_bin();
    let model = sir_basic();
    let m = model.to_str().unwrap();
    // Integer backends reconcile exactly; the ODE's real-valued flows against
    // the rounded S column reconcile within a count.
    for (backend, extra, tol) in [
        ("chain_binomial", vec!["--dt", "1"], 0.0),
        ("gillespie", vec![], 0.0),
        ("ode", vec!["--dt", "1"], 1.0),
    ] {
        let mut args = vec!["simulate", m, "--scenario", "baseline", "--seed", "11",
            "--backend", backend, "--stdout"];
        args.extend(extra);
        let stdout = run(&camdl, &args);
        assert_periods_tile_and_reconcile(backend, &stdout, tol);
    }
}

/// `--output-every` changes which instants are written; the periods still
/// tile between the rows that are, and the flows still reconcile — so a
/// coarser cadence writes wider periods, not one-day flows sampled every N.
#[test]
fn a_coarser_output_cadence_widens_the_periods_not_the_sampling() {
    let camdl = camdl_bin();
    let model = sir_basic();
    let stdout = run(&camdl, &["simulate", model.to_str().unwrap(), "--scenario", "baseline",
        "--seed", "11", "--backend", "chain_binomial", "--dt", "1", "--output-every", "10", "--stdout"]);
    assert_periods_tile_and_reconcile("chain_binomial --output-every 10", &stdout, 0.0);
    let (header, rows) = table(&stdout);
    let (start, stop) = (col(&header, "t_start"), col(&header, "t_stop"));
    assert!(rows[1..].iter().all(|r| (num(r, stop) - num(r, start) - 10.0).abs() < 1e-9),
        "every period after the first is ten days wide");
}

/// Under `--dates` the three boundaries get their calendar twins, rendered by
/// the same map as `date`: `date_stop` is the row's `date`, `date_start` the
/// previous row's.
#[test]
fn dates_gives_each_boundary_its_calendar_twin() {
    let camdl = camdl_bin();
    let model = seed_timing_dated();
    let stdout = run(&camdl, &["simulate", model.to_str().unwrap(), "--backend", "chain_binomial",
        "--dt", "1", "--seed", "3", "--dates", "--stdout",
        "--param", "beta=0.6", "--param", "gamma=0.2", "--param", "lambda=2.0", "--param", "w=3.0",
        "--param", "N0=5000", "--param", "rho=0.5", "--param", "k=20", "--param", "tau=2"]);
    let (header, rows) = table(&stdout);
    assert_eq!(&header[..6], ["t", "t_start", "t_stop", "date", "date_start", "date_stop"]);
    let (date, ds, de) = (col(&header, "date"), col(&header, "date_start"), col(&header, "date_stop"));
    assert_eq!(rows[0][date], "2020-02-24", "the origin");
    assert_eq!(rows[0][ds], rows[0][date], "the first row's empty period opens on its own date");
    assert_eq!(rows[0][de], rows[0][date]);
    for w in rows.windows(2) {
        let (prev, row) = (&w[0], &w[1]);
        assert_eq!(row[de], row[date], "date_stop is the row's date");
        assert_eq!(row[ds], prev[date], "date_start is the previous row's date");
    }
    assert_eq!(rows[1][ds], "2020-02-24");
    assert_eq!(rows[1][de], "2020-02-25");
}
