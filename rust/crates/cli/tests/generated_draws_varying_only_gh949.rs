//! gh#949: a generated draw row (`simulate --draws prior|uniform`) carries only
//! the parameters the measure varies — a prior's parameters, or the bounded
//! parameters a uniform sweep draws. Every other parameter resolves per cell
//! through the ordinary tiers: model default → a `--fit` config's `[fixed]` →
//! `--params` → the cell's scenario → `--param`.
//!
//! The defect this pins: the rows used to carry every parameter. A row
//! resolves at the draw tier, above `--params`, so
//!
//! - `--draws prior --scenario half,other` filled the no-prior parameter from
//!   the listed scenarios applied in order, so arm `other` ran at `half`'s
//!   value and the two arms produced identical bands (a zero counterfactual
//!   effect, silently);
//! - a `--params` value for a no-prior (or unbounded) parameter was ignored;
//! - `--draws-out` exported the leaked values, differently for each scenario
//!   order.
//!
//! CLI-level on purpose: the leak sat between the generator and the resolver's
//! tier order, which neither unit alone shows.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn skip_if_missing_binary() -> PathBuf {
    let bin = binary();
    assert!(
        bin.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test` (gh#105)",
        bin.display()
    );
    bin
}

/// `beta`/`gamma` carry priors (and bounds); `c` has neither and no default.
/// Only `half` sets `c`.
const TWO_ARM: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate in [0.05, 1.0] ~ log_normal(mu = -1.2, sigma = 0.5)
  gamma : rate in [0.02, 0.5] ~ log_normal(mu = -2.3, sigma = 0.3)
  c     : probability
}
init { S = 990  I = 10  R = 0 }
transitions {
  infect  : S --> I  @ (1 - c) * beta * S * I / (S + I + R)
  recover : I --> R  @ gamma * I
}
simulate { from = 0 'days  to = 100 'days }
scenarios {
  half   { set = { c = 0.5 } }
  other  { enable = [] }
}
quantities { final_R = final(R) }
"#;

/// Both arms set `c`; the no-prior parameter the old generator filled from the
/// LAST listed scenario.
const VAX: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate in [0.05, 1.0] ~ log_normal(mu = -1.2, sigma = 0.5)
  gamma : rate in [0.02, 0.5] ~ log_normal(mu = -2.3, sigma = 0.3)
  c     : probability
}
init { S = 990  I = 10  R = 0 }
transitions {
  infect  : S --> I  @ (1 - c) * beta * S * I / (S + I + R)
  recover : I --> R  @ gamma * I
}
simulate { from = 0 'days  to = 100 'days }
scenarios {
  novax { set = { c = 0.0 } }
  vax   { set = { c = 0.5 } }
}
quantities { final_R = final(R) }
"#;

/// `c` has a model default (a typed constant `let` is a fixed parameter) and no
/// bounds, so `--draws uniform` does not draw it; `beta`/`gamma` are bounded
/// and are drawn.
const UNIFORM: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate in [0.05, 1.0]
  gamma : rate in [0.02, 0.5]
}
let c : probability = 0.0
init { S = 990  I = 10  R = 0 }
transitions {
  infect  : S --> I  @ (1 - c) * beta * S * I / (S + I + R)
  recover : I --> R  @ gamma * I
}
simulate { from = 0 'days  to = 100 'days }
quantities { final_R = final(R) }
"#;

struct Fixture {
    dir: tempfile::TempDir,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        std::fs::write(p.join("two_arm.camdl"), TWO_ARM).unwrap();
        std::fs::write(p.join("vax.camdl"), VAX).unwrap();
        std::fs::write(p.join("uniform.camdl"), UNIFORM).unwrap();
        std::fs::write(p.join("b0.toml"), "c = 0.0\n").unwrap();
        std::fs::write(p.join("c09.toml"), "c = 0.9\n").unwrap();
        Fixture { dir }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.dir.path().join(name)
    }

    fn camdl(&self, args: &[&str]) -> Output {
        Command::new(skip_if_missing_binary())
            .args(args)
            .current_dir(self.dir.path())
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .output()
            .expect("spawn camdl")
    }

    fn ok(&self, args: &[&str]) {
        let out = self.camdl(args);
        assert!(
            out.status.success(),
            "camdl {:?} failed:\n{}",
            args,
            String::from_utf8_lossy(&out.stderr)
        );
    }

    fn read(&self, name: &str) -> String {
        std::fs::read_to_string(self.path(name))
            .unwrap_or_else(|e| panic!("read {name}: {e}"))
    }

    /// `scenario -> the quantile columns` of `<dir>/quantities/final_R.tsv`.
    /// A run with no `--scenario` has no scenario column; its one row is keyed
    /// `""`.
    fn final_r_rows(&self, dir: &str) -> Vec<(String, String)> {
        let tsv = self.read(&format!("{dir}/quantities/final_R.tsv"));
        let mut lines = tsv.lines();
        let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
        let has_scenario = header.first() == Some(&"scenario");
        lines
            .map(|l| {
                let cols: Vec<&str> = l.split('\t').collect();
                if has_scenario {
                    (cols[0].to_string(), cols[1..].join("\t"))
                } else {
                    (String::new(), cols.join("\t"))
                }
            })
            .collect()
    }
}

fn row<'a>(rows: &'a [(String, String)], scenario: &str) -> &'a str {
    rows.iter()
        .find(|(s, _)| s == scenario)
        .map(|(_, v)| v.as_str())
        .unwrap_or_else(|| panic!("no row for scenario '{scenario}' in {rows:?}"))
}

fn header(tsv: &str) -> Vec<String> {
    tsv.lines().next().unwrap_or("").split('\t').map(str::to_string).collect()
}

/// Test 1. The arm that does not set `c` must run at the `--params` value
/// (c = 0), not at the value the other arm sets — and must equal a run of that
/// arm alone.
#[test]
fn an_arm_that_sets_nothing_does_not_inherit_another_arms_value() {
    let f = Fixture::new();
    f.ok(&[
        "simulate", "two_arm.camdl", "--params", "b0.toml", "--draws", "prior", "-n", "3",
        "--scenario", "half,other", "--quantities-out", "q_both",
    ]);
    f.ok(&[
        "simulate", "two_arm.camdl", "--params", "b0.toml", "--draws", "prior", "-n", "3",
        "--scenario", "other", "--quantities-out", "q_other",
    ]);
    let both = f.final_r_rows("q_both");
    let alone = f.final_r_rows("q_other");
    assert_ne!(
        row(&both, "half"),
        row(&both, "other"),
        "half (c = 0.5) and other (c = 0 from --params) must differ: {both:?}"
    );
    assert_eq!(
        row(&both, "other"),
        row(&alone, "other"),
        "arm `other` must not depend on which other arms were listed"
    );
}

/// Test 2. `--params` supplies the no-prior parameter.
#[test]
fn params_supplies_a_parameter_without_a_prior() {
    let f = Fixture::new();
    f.ok(&[
        "simulate", "two_arm.camdl", "--params", "b0.toml", "--draws", "prior", "-n", "3",
        "--scenario", "other",
    ]);
}

/// Test 3. The exported draws are a function of the prior alone: identical
/// across scenario orders, with no column for the unsampled `c`.
#[test]
fn draws_out_is_independent_of_scenario_order_and_omits_unsampled_columns() {
    let f = Fixture::new();
    f.ok(&[
        "simulate", "vax.camdl", "--draws", "prior", "-n", "3", "--scenario", "novax,vax",
        "--draws-out", "a.tsv",
    ]);
    f.ok(&[
        "simulate", "vax.camdl", "--draws", "prior", "-n", "3", "--scenario", "vax,novax",
        "--draws-out", "b.tsv",
    ]);
    let (a, b) = (f.read("a.tsv"), f.read("b.tsv"));
    assert_eq!(a, b, "--draws-out must not depend on scenario order");
    assert_eq!(header(&a), vec!["beta", "gamma"], "no column for the unsampled `c`: {a}");
}

/// Test 4. `--draws uniform` leaves the unbounded `c` to `--params`: the run
/// equals the same draws with `c` pinned by `--param` (tier 5, which always
/// won), and `--draws-out` has no `c` column.
#[test]
fn uniform_draws_leave_an_unbounded_parameter_to_params() {
    let f = Fixture::new();
    f.ok(&[
        "simulate", "uniform.camdl", "--params", "c09.toml", "--draws", "uniform", "-n", "3",
        "--quantities-out", "q_params", "--draws-out", "u.tsv",
    ]);
    f.ok(&[
        "simulate", "uniform.camdl", "--param", "c=0.9", "--draws", "uniform", "-n", "3",
        "--quantities-out", "q_param",
    ]);
    f.ok(&[
        "simulate", "uniform.camdl", "--draws", "uniform", "-n", "3",
        "--quantities-out", "q_default",
    ]);
    let via_params = f.final_r_rows("q_params");
    assert_eq!(
        via_params,
        f.final_r_rows("q_param"),
        "the --params value c = 0.9 must be the one that ran"
    );
    assert_ne!(
        via_params,
        f.final_r_rows("q_default"),
        "c = 0.9 and the model default c = 0 must give different bands"
    );
    assert_eq!(header(&f.read("u.tsv")), vec!["beta", "gamma"]);
}

/// Test 5 (negative control). The sampled parameters still vary across draws
/// and are exported, for both measures. Deliberately written to pass before
/// the fix too (an invocation the old code accepted, and no assertion on the
/// absent columns — tests 3 and 4 own those): the fix must not stop a
/// measure's own parameters from varying.
#[test]
fn sampled_parameters_still_vary_and_are_exported() {
    let f = Fixture::new();
    f.ok(&[
        "simulate", "vax.camdl", "--draws", "prior", "-n", "4", "--scenario", "novax",
        "--draws-out", "p.tsv",
    ]);
    f.ok(&[
        "simulate", "uniform.camdl", "--draws", "uniform", "-n", "4", "--draws-out", "u.tsv",
    ]);
    for file in ["p.tsv", "u.tsv"] {
        let tsv = f.read(file);
        let cols = header(&tsv);
        let rows: Vec<Vec<&str>> = tsv.lines().skip(1).map(|l| l.split('\t').collect()).collect();
        assert_eq!(rows.len(), 4, "{file}: one row per draw");
        for name in ["beta", "gamma"] {
            let j = cols.iter().position(|c| c == name)
                .unwrap_or_else(|| panic!("{file}: sampled `{name}` must be exported: {tsv}"));
            let mut vals: Vec<&str> = rows.iter().map(|r| r[j]).collect();
            vals.sort();
            vals.dedup();
            assert_eq!(vals.len(), 4, "{file}: `{name}` must differ across draws: {tsv}");
        }
    }
}

/// The refusal fires per arm, before anything runs: `half` sets `c`, `other`
/// does not and nothing else does, so the run is refused naming `other` (and
/// only `other`), and no draws file or store leaf is written.
#[test]
fn an_arm_with_no_value_for_an_unsampled_parameter_is_refused_up_front() {
    let f = Fixture::new();
    let out = f.camdl(&[
        "simulate", "two_arm.camdl", "--draws", "prior", "-n", "3", "--scenario", "half,other",
        "--draws-out", "d.tsv",
    ]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "must refuse:\n{stderr}");
    assert!(
        stderr.contains(
            "parameter 'c' has no prior and no default value, and nothing sets it for scenario 'other'"
        ),
        "must name the parameter and the arm:\n{stderr}"
    );
    assert!(!stderr.contains("scenario 'half'"), "half sets c and must not be refused:\n{stderr}");
    assert!(!f.path("d.tsv").exists(), "a refused run writes no --draws-out");
    assert!(!f.path("results").exists(), "a refused run commits no store leaf");
}

/// A `--fit` config's `[fixed]` block resolves at its own tier, below
/// `--params`: `--params c = 0` wins over `[fixed] c = 0.5`, and the exported
/// rows carry only the `[estimate]` parameters.
#[test]
fn fit_fixed_resolves_below_params() {
    let f = Fixture::new();
    let fit = |name: &str, c: f64| {
        std::fs::write(
            f.path(name),
            format!(
                "[model]\ncamdl = \"two_arm.camdl\"\n\n[estimate]\n\
                 beta  = {{ bounds = [0.05, 1.0] }}\ngamma = {{ bounds = [0.02, 0.5] }}\n\n\
                 [fixed]\nc = {c}\n"
            ),
        )
        .unwrap();
    };
    fit("fit_c05.toml", 0.5);
    fit("fit_c0.toml", 0.0);
    f.ok(&[
        "simulate", "two_arm.camdl", "--fit", "fit_c05.toml", "--params", "b0.toml",
        "--draws", "prior", "-n", "3", "--quantities-out", "q_override", "--draws-out", "d.tsv",
    ]);
    f.ok(&[
        "simulate", "two_arm.camdl", "--fit", "fit_c0.toml", "--draws", "prior", "-n", "3",
        "--quantities-out", "q_c0",
    ]);
    assert_eq!(
        f.final_r_rows("q_override"),
        f.final_r_rows("q_c0"),
        "--params c = 0 must win over the fit's [fixed] c = 0.5"
    );
    assert_eq!(header(&f.read("d.tsv")), vec!["beta", "gamma"]);
}
