//! `camdl design lhs MODEL --fit FIT.toml --n N --seed S [-o FILE]`, end to
//! end: the command writes a Latin-hypercube design that `camdl simulate
//! --sweep-file` runs unchanged, it is a function of its four inputs, it
//! writes nothing but the design, and it refuses the two inputs it cannot
//! stratify.
//!
//! Shells out to the release `camdl` binary. The sampler's own properties
//! (bit equality with `build_lhs_chain_starts`, one point per bin on each
//! parameter's scale) are unit tests in `cli/src/design.rs`.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let p = Path::new(&manifest).join("../../target/release/camdl");
    assert!(p.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test`",
        p.display());
    p
}

fn camdlc_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(p.exists(), "camdlc.exe missing: {} - run `make build`", p.display());
    p
}

/// `beta` is a rate with a positive lower bound, so its box is sampled on the
/// log scale; `rho` is a probability, sampled on the linear scale. `mu` has no
/// declared range: its search box is `[0, inf)`.
const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.001, 5.0]
  gamma : rate  in [0.01, 1.0]
  rho   : probability
  mu    : rate
  N0    : count in [100, 10000]
}
transitions {
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
  death     : I --> @ mu * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected  = rho * I
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 999  I = 1 }
simulate { from = 0 'days  to = 6 'days }
"#;

struct Fixture {
    dir: tempfile::TempDir,
    ir: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("sir.camdl");
        std::fs::write(&model, MODEL).unwrap();
        let out = Command::new(camdlc_bin()).env_remove("CLICOLOR_FORCE").arg(&model).output().unwrap();
        assert!(out.status.success(), "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
        let ir = dir.path().join("sir.ir.json");
        std::fs::write(&ir, &out.stdout).unwrap();
        std::fs::write(dir.path().join("cases.tsv"),
            "time\tcases\n1\t2\n2\t4\n3\t8\n4\t6\n5\t4\n6\t2\n").unwrap();
        Fixture { dir, ir }
    }

    fn path(&self, name: &str) -> PathBuf { self.dir.path().join(name) }

    /// A fit config estimating `estimate` (the body of its `[estimate]`
    /// table) with `mu` and `N0` fixed unless `estimate` names them.
    fn fit_toml(&self, name: &str, estimate: &str, fixed: &str) -> String {
        let p = self.path(name);
        std::fs::write(&p, format!(r#"
[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
[config]
dt = 1.0
[estimate]
{estimate}
[fixed]
{fixed}
[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 1
particles = 30
iterations = 5
cooling = 0.9
"#, ir = self.ir.display(), data = self.path("cases.tsv").display())).unwrap();
        p.to_string_lossy().into_owned()
    }

    /// `rho` before `beta` before `gamma`: the table's order is not name order.
    fn bounded_fit(&self) -> String {
        self.fit_toml("fit.toml",
            "rho   = { bounds = [0.1, 0.9] }\n\
             beta  = { bounds = [0.01, 5.0], start = 0.4 }\n\
             gamma = { start = 0.15 }",
            "mu = 0.0\nN0 = 1000")
    }

    /// Run `camdl <args>` with the fixture's directory as the working
    /// directory, so a store the command wrote by default would land in it.
    fn camdl(&self, args: &[&str]) -> Output {
        Command::new(camdl_bin())
            .env_remove("CLICOLOR_FORCE")
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .current_dir(self.dir.path())
            .args(args)
            .output()
            .expect("camdl must invoke")
    }

    fn design(&self, fit: &str, extra: &[&str]) -> Output {
        let ir = self.ir.to_string_lossy().into_owned();
        let mut args = vec!["design", "lhs", ir.as_str(), "--fit", fit];
        args.extend_from_slice(extra);
        self.camdl(&args)
    }

    /// The names in the fixture's directory, sorted.
    fn listing(&self) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(self.dir.path()).unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }
}

fn assert_ok(out: &Output, what: &str) {
    assert!(out.status.success(), "{what} must exit 0.\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr));
}

/// A TSV's header and rows, without the `#` lines `simulate -o` opens with.
fn table(text: &str) -> (Vec<String>, Vec<Vec<String>>) {
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    let split = |l: &str| l.split('\t').map(str::to_string).collect::<Vec<_>>();
    let header = split(lines.next().expect("a header"));
    (header, lines.map(split).collect())
}

/// The design `design lhs` writes is run by `simulate --sweep-file` as it
/// stands: one design point per row, under the row's `point_id`, at the row's
/// values.
#[test]
fn the_design_runs_under_simulate_sweep_file() {
    let fx = Fixture::new();
    let fit = fx.bounded_fit();
    let design_path = fx.path("design.tsv").to_string_lossy().into_owned();
    assert_ok(&fx.design(&fit, &["--n", "8", "--seed", "1", "-o", &design_path]), "design lhs");

    let design = std::fs::read_to_string(&design_path).unwrap();
    assert!(!design.contains('#'), "the design carries no comment line");
    let (header, rows) = table(&design);
    assert_eq!(header, ["point_id", "rho", "beta", "gamma"],
        "point_id, then the parameters in the [estimate] table's order");
    assert_eq!(rows.len(), 8);
    for (i, row) in rows.iter().enumerate() {
        assert_eq!(row[0], i.to_string(), "point_id is the 0-based row");
        let value = |col: usize| row[col].parse::<f64>().unwrap();
        assert!((0.1..=0.9).contains(&value(1)), "rho within its [estimate] bounds: {row:?}");
        assert!((0.01..=5.0).contains(&value(2)), "beta within its [estimate] bounds: {row:?}");
        assert!((0.01..=1.0).contains(&value(3)), "gamma within the model's range: {row:?}");
    }

    let ir = fx.ir.to_string_lossy().into_owned();
    let params = fx.path("params.toml");
    std::fs::write(&params, "mu = 0.0\nN0 = 1000\n").unwrap();
    let combined = fx.path("combined.tsv").to_string_lossy().into_owned();
    let store = fx.path("store").to_string_lossy().into_owned();
    assert_ok(&fx.camdl(&[
        "simulate", &ir, "--params", &params.to_string_lossy(),
        "--sweep-file", &design_path, "--seed", "3",
        "--backend", "chain_binomial", "--dt", "1", "--output-dir", &store, "-o", &combined,
    ]), "simulate --sweep-file");

    // The combined table carries each point's id and its swept values, as
    // written in the design.
    let (header, sim_rows) = table(&std::fs::read_to_string(&combined).unwrap());
    let col = |name: &str| header.iter().position(|h| h == name)
        .unwrap_or_else(|| panic!("no `{name}` column in {header:?}"));
    let (pid, rho, beta, gamma) =
        (col("point_id"), col("sweep:rho"), col("sweep:beta"), col("sweep:gamma"));
    let mut seen: std::collections::BTreeMap<String, Vec<f64>> = Default::default();
    for r in &sim_rows {
        let values = [rho, beta, gamma].map(|c| r[c].parse::<f64>().unwrap()).to_vec();
        seen.insert(r[pid].clone(), values);
    }
    assert_eq!(seen.len(), 8, "eight design points ran: {:?}", seen.keys());
    for row in &rows {
        let written: Vec<f64> = row[1..].iter().map(|v| v.parse().unwrap()).collect();
        assert_eq!(seen[&row[0]], written, "point {} ran at the design's values", row[0]);
    }
}

/// The same model, fit config, `--n` and `--seed` give the same bytes, on
/// stdout as in the `-o` file; another seed gives other bytes. `--seed`
/// defaults to 1.
#[test]
fn the_design_is_a_function_of_its_inputs() {
    let fx = Fixture::new();
    let fit = fx.bounded_fit();
    let first = fx.design(&fit, &["--n", "20", "--seed", "1"]);
    assert_ok(&first, "design lhs");
    assert!(!first.stdout.is_empty());
    assert_eq!(fx.design(&fit, &["--n", "20", "--seed", "1"]).stdout, first.stdout,
        "a second run writes the same bytes");
    assert_eq!(fx.design(&fit, &["--n", "20"]).stdout, first.stdout, "--seed defaults to 1");

    let file = fx.path("d.tsv");
    assert_ok(&fx.design(&fit, &["--n", "20", "--seed", "1", "-o", &file.to_string_lossy()]), "-o");
    assert_eq!(std::fs::read(&file).unwrap(), first.stdout, "-o writes what stdout carries");

    let other = fx.design(&fit, &["--n", "20", "--seed", "2"]);
    assert_ok(&other, "design lhs --seed 2");
    assert_ne!(other.stdout, first.stdout, "another seed gives another design");
}

/// The command writes the design and nothing else: no store, no cache entry
/// beside the inputs.
#[test]
fn nothing_but_the_design_is_written() {
    let fx = Fixture::new();
    let fit = fx.bounded_fit();
    let before = fx.listing();
    assert_ok(&fx.design(&fit, &["--n", "5", "--seed", "1"]), "design lhs to stdout");
    assert_eq!(fx.listing(), before, "writing to stdout leaves the directory as it was");
    assert_ok(&fx.design(&fit, &["--n", "5", "--seed", "1", "-o", "d.tsv"]), "design lhs -o");
    let mut expected = before;
    expected.push("d.tsv".into());
    expected.sort();
    assert_eq!(fx.listing(), expected, "-o adds the design file only");
}

#[test]
fn fewer_than_two_points_is_refused() {
    let fx = Fixture::new();
    let fit = fx.bounded_fit();
    let out = fx.design(&fit, &["--n", "1", "--seed", "1"]);
    assert!(!out.status.success());
    assert!(out.stdout.is_empty(), "a refused design writes no rows");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("--n must be at least 2 (got 1)"), "{stderr}");
}

/// `mu` is declared without a range and the fit gives it no bounds, so its
/// search box is `[0, inf)` and there are no equal-width bins to cut.
#[test]
fn an_estimated_parameter_without_a_finite_box_is_refused_by_name() {
    let fx = Fixture::new();
    let fit = fx.fit_toml("open.toml",
        "beta = { bounds = [0.01, 5.0], start = 0.4 }\nmu = { start = 0.1 }",
        "rho = 0.5\ngamma = 0.2\nN0 = 1000");
    let out = fx.design(&fit, &["--n", "10", "--seed", "1"]);
    assert!(!out.status.success());
    assert!(out.stdout.is_empty(), "a refused design writes no rows");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("these parameters have no finite one: `mu` [0, inf]"), "{stderr}");
    assert!(stderr.contains("bounds = [lo, hi]"), "{stderr}");
}
