//! gh#572: a scenario that touches a parameter a user-authored coordinate
//! varies — a `--sweep` / `--sweep-file` point set, or a `--draws <file.tsv>`
//! — is refused before any cell runs (proposal 2026-06-27 §4).
//!
//! Precedence would let the scenario win: its `set` overrides, and its `scale`
//! rescales, every swept value, so the grid would not vary that parameter and
//! the store would hold leaves labelled with values that never ran. The
//! refusal covers every command that plans cells: `simulate --sweep`,
//! `simulate --sweep-file`, `simulate --draws <file>`, and the `--dry-run` of
//! each. Generated draws (`--draws uniform|prior|posterior`) are the negative
//! control: a scenario over generated draws is the counterfactual those runs
//! exist for, and stays allowed.
//!
//! CLI-level on purpose: the defect class is a guard that some command path
//! does not reach, which a test of the guard function alone cannot see.

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

/// Two death-like rates, `mu` and `nu`, both in the dynamics. The presets
/// touch them in each of the ways the guard has to see:
///
/// - `pin_mu`: `set mu` directly.
/// - `half_mu`: `scale mu` directly.
/// - `combo`: sets only `nu` itself, and gets `mu` from the composed `pin_mu`.
/// - `pin_nu`: touches only `nu`, which no test sweeps (the allowed case).
const MODEL: &str = r#"
time_unit = 'days
compartments { S, I }
parameters {
  mu : rate in [0.001, 10.0]
  nu : rate in [0.001, 10.0]
}
init { S = 1000  I = 0 }
transitions {
  death : S --> I  @ mu * S
  clear : I -->    @ nu * I
}
simulate { from = 0 'days  to = 20 'days }
scenarios {
  pin_mu  { set = { mu = 0.3 } }
  half_mu { scale = { mu = 0.5 } }
  pin_nu  { set = { nu = 0.2 } }
  combo   { compose = [pin_mu]  set = { nu = 0.2 } }
}
"#;

/// A scratch directory holding the model and a params file; every run sweeps
/// `mu` over three points under one scenario.
struct Fixture {
    dir: tempfile::TempDir,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        std::fs::write(p.join("m.camdl"), MODEL).unwrap();
        std::fs::write(p.join("p.toml"), "mu = 0.1\nnu = 0.1\n").unwrap();
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

    /// `simulate m.camdl --params p.toml --backend ode --output-dir out --sweep
    /// mu=0.2,0.3,0.4 --scenario <scenario> <flags…>` — a three-point `mu`
    /// sweep on `simulate`'s design axis.
    fn simulate_sweep(&self, scenario: &str, flags: &[&str]) -> Output {
        let mut args = vec![
            "simulate", "m.camdl", "--params", "p.toml", "--backend", "ode",
            "--output-dir", "out", "--sweep", "mu=0.2,0.3,0.4", "--scenario", scenario,
        ];
        args.extend_from_slice(flags);
        self.camdl(&args)
    }

    /// Every `Sim` leaf under the output root (`out/sims/`; a multi-cell
    /// `simulate` also writes an ensemble leaf under `out/ensembles/`, which
    /// is not a computation).
    fn leaves(&self) -> Vec<PathBuf> {
        run_leaves(&self.path("out").join("sims"))
    }
}

fn run_leaves(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if dir.join("run.json").is_file() {
            out.push(dir.clone());
        }
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for e in entries.flatten() {
                if e.path().is_dir() {
                    stack.push(e.path());
                }
            }
        }
    }
    out
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// Assert `out` failed with a diagnostic containing every line in `lines`.
fn assert_refused(out: &Output, what: &str, lines: &[&str]) {
    let err = stderr(out);
    assert!(!out.status.success(), "{what}: must be refused, but exited 0; stderr:\n{err}");
    for line in lines {
        assert!(err.contains(line), "{what}: diagnostic lacks {line:?}; stderr:\n{err}");
    }
}

/// `simulate --sweep` and `simulate --sweep --dry-run` both refuse the grid
/// with the same diagnostic, and nothing is written to the store (proposal
/// 2026-10-24-batch-to-simulate §6 test 5).
fn assert_simulate_sweep_refuses(scenario: &str, lines: &[&str]) {
    let fx = Fixture::new();
    for flags in [&["--dry-run"][..], &[][..]] {
        let what = format!("{scenario} / simulate --sweep {}", flags.join(" "));
        assert_refused(&fx.simulate_sweep(scenario, flags), &what, lines);
    }
    assert!(fx.leaves().is_empty(), "{scenario}: a refused sweep must write no leaf");
}

#[test]
fn sweep_of_a_parameter_the_scenario_sets_is_refused() {
    assert_simulate_sweep_refuses(
        "pin_mu",
        &[
            "error: parameter `mu` is controlled by both the sweep and scenario `pin_mu`",
            "  sweep:     mu = [0.2, 0.3, 0.4]",
            "  scenario:  set mu = 0.3",
            "The scenario would override every sweep value, so this sweep would not vary `mu`.",
            "Fix: remove `mu` from the sweep, or use a scenario that does not touch it.",
        ],
    );
}

#[test]
fn sweep_of_a_parameter_the_scenario_scales_is_refused() {
    assert_simulate_sweep_refuses(
        "half_mu",
        &[
            "error: parameter `mu` is controlled by both the sweep and scenario `half_mu`",
            "  sweep:     mu = [0.2, 0.3, 0.4]",
            "  scenario:  scale mu \u{d7} 0.5",
            "The scenario would rescale every sweep point of `mu`; composing a sweep with a \
             scenario that scales the same parameter is not supported.",
            "Fix: remove `mu` from the sweep, or use a scenario that does not touch it.",
        ],
    );
}

#[test]
fn sweep_of_a_parameter_a_composed_preset_sets_is_refused() {
    // `combo` itself sets only `nu`; `mu` arrives through `compose = [pin_mu]`.
    assert_simulate_sweep_refuses(
        "combo",
        &[
            "error: parameter `mu` is controlled by both the sweep and scenario `combo`",
            "  scenario:  set mu = 0.3    (from composed preset `pin_mu`)",
            "The scenario would override every sweep value, so this sweep would not vary `mu`.",
        ],
    );
}

#[test]
fn sweep_under_a_scenario_on_another_parameter_runs_every_point() {
    // `pin_nu` touches only `nu`: no collision, three distinct leaves.
    // (cas_integration.rs `sweep_point_is_recorded_in_run_json_and_listed`
    // covers the same allowed shape on sir_basic under the implicit arm.)
    let fx = Fixture::new();
    let dry = fx.simulate_sweep("pin_nu", &["--dry-run"]);
    assert!(dry.status.success(), "dry run must plan the grid; stderr:\n{}", stderr(&dry));
    assert!(
        stderr(&dry).contains("0 cache hits → read back, 3 misses → would simulate"),
        "the plan is three cells, none stored yet; stderr:\n{}",
        stderr(&dry)
    );
    assert!(fx.leaves().is_empty(), "a dry run writes nothing");
    let run = fx.simulate_sweep("pin_nu", &[]);
    assert!(run.status.success(), "simulate --sweep must succeed; stderr:\n{}", stderr(&run));
    assert_eq!(fx.leaves().len(), 3, "three sweep points are three computations");
    // The same grid again: every cell is a hit, planned and run alike, and no
    // leaf is added.
    let dry = fx.simulate_sweep("pin_nu", &["--dry-run"]);
    assert!(dry.status.success(), "second dry run must plan; stderr:\n{}", stderr(&dry));
    assert!(
        stderr(&dry).contains("3 cache hits → read back, 0 misses"),
        "the dry run counts the three leaves; stderr:\n{}",
        stderr(&dry)
    );
    let again = fx.simulate_sweep("pin_nu", &[]);
    assert!(again.status.success(), "second run must succeed; stderr:\n{}", stderr(&again));
    assert!(
        stderr(&again).contains("3 of 3 cells read from the store; simulated 0"),
        "the second run reads every cell back; stderr:\n{}",
        stderr(&again)
    );
    assert_eq!(fx.leaves().len(), 3, "the same grid adds no leaf");
}

#[test]
fn sweep_file_over_a_parameter_the_scenario_sets_is_refused() {
    // A `--sweep-file` row set runs as a sweep, so the same refusal applies,
    // on the run and on the dry run, before anything is written.
    let fx = Fixture::new();
    std::fs::write(fx.path("mu.tsv"), "mu\n0.2\n0.3\n0.4\n").unwrap();
    for extra in [&["--dry-run"][..], &[][..]] {
        let mut args = vec![
            "simulate", "m.camdl", "--params", "p.toml", "--backend", "ode",
            "--output-dir", "out", "--sweep-file", "mu.tsv", "--scenario", "pin_mu",
        ];
        args.extend_from_slice(extra);
        assert_refused(
            &fx.camdl(&args),
            "sweep file × pin_mu",
            &[
                "error: parameter `mu` is controlled by both the sweep and scenario `pin_mu`",
                "  sweep:     mu = [0.2, 0.3, 0.4]",
            ],
        );
    }
    assert!(fx.leaves().is_empty(), "a refused sweep file writes no leaf");
}

#[test]
fn draws_file_column_the_scenario_sets_is_refused_with_the_shared_diagnostic() {
    let fx = Fixture::new();
    std::fs::write(fx.path("mydraws.tsv"), "mu\tnu\n0.3\t0.5\n0.4\t0.5\n").unwrap();
    let out = fx.camdl(&[
        "simulate", "m.camdl", "--draws", "mydraws.tsv", "--scenario", "pin_mu",
        "--backend", "ode", "--output-dir", "out", "-o", "traj.tsv",
    ]);
    assert_refused(
        &out,
        "draws file × pin_mu",
        &[
            "error: parameter `mu` is controlled by both the draws file `mydraws.tsv` and \
             scenario `pin_mu`",
            "  draws file:  mu = [0.3, 0.4]",
            "  scenario:    set mu = 0.3",
            "The scenario would override the file's `mu` in every draw, so that column would \
             have no effect.",
            "Fix: drop the `mu` column from the draws file, or use a scenario that does not \
             touch it.",
        ],
    );
    assert!(fx.leaves().is_empty(), "a refused draws file must write no leaf");
}

#[test]
fn simulate_dry_run_refuses_a_draws_file_column_the_scenario_sets() {
    // The dry run plans the same cells the run would, so it refuses the same
    // collision — before printing a plan.
    let fx = Fixture::new();
    std::fs::write(fx.path("mydraws.tsv"), "mu\tnu\n0.3\t0.5\n0.4\t0.5\n").unwrap();
    let out = fx.camdl(&[
        "simulate", "m.camdl", "--draws", "mydraws.tsv", "--scenario", "pin_mu",
        "--backend", "ode", "--output-dir", "out", "-o", "traj.tsv", "--dry-run",
    ]);
    assert_refused(
        &out,
        "simulate --dry-run × draws file × pin_mu",
        &[
            "error: parameter `mu` is controlled by both the draws file `mydraws.tsv` and \
             scenario `pin_mu`",
            "Fix: drop the `mu` column from the draws file, or use a scenario that does not \
             touch it.",
        ],
    );
    assert!(
        !stderr(&out).contains("(dry run)"),
        "the refusal comes before the plan is printed; stderr:\n{}",
        stderr(&out)
    );
}

#[test]
fn draws_file_column_the_scenario_scales_is_refused() {
    let fx = Fixture::new();
    std::fs::write(fx.path("mydraws.tsv"), "mu\tnu\n0.3\t0.5\n0.4\t0.5\n").unwrap();
    let out = fx.camdl(&[
        "simulate", "m.camdl", "--draws", "mydraws.tsv", "--scenario", "half_mu",
        "--backend", "ode", "--output-dir", "out", "-o", "traj.tsv",
    ]);
    assert_refused(
        &out,
        "draws file × half_mu",
        &[
            "  scenario:    scale mu \u{d7} 0.5",
            "The scenario would rescale the file's `mu` in every draw; composing a draws file \
             with a scenario that scales the same parameter is not supported.",
        ],
    );
}

/// The negative control, and the identity invariant PR #941 pinned. Generated
/// draws under a scenario that sets the parameter they vary stay allowed. On a
/// one-parameter model with one explicit seed, all three draws resolve to the
/// same θ (`mu = 0.01`, the scenario's) and the same seed: one computation,
/// so one leaf, labelled with the value that ran. Without the scenario the
/// three draws are three computations.
///
/// This is where the "a cell's `params` level is the parameters the engine
/// ran" invariant (gh#583) is still reachable end to end: a sweep under such
/// a scenario is refused by this issue's guard.
#[test]
fn generated_draws_under_a_scenario_are_allowed_and_key_the_values_that_ran() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    std::fs::write(
        p.join("pd.camdl"),
        r#"
time_unit = 'days
compartments { S, D }
parameters { mu : rate in [0.001, 5.0] }
transitions { death : S --> D @ mu * S }
init { S = 1000  D = 0 }
simulate { from = 0 'days  to = 20 'days }
scenarios { slow { set = { mu = 0.01 } } }
"#,
    )
    .unwrap();
    let bin = skip_if_missing_binary();
    let simulate = |out: &str, scenario: Option<&str>| {
        let mut cmd = Command::new(&bin);
        cmd.current_dir(p)
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .args([
                "simulate", "pd.camdl", "--draws", "uniform", "-n", "3", "--seeds", "1",
                "--backend", "ode", "--output-dir", out, "-o", &format!("{out}.tsv"),
            ]);
        if let Some(s) = scenario {
            cmd.args(["--scenario", s]);
        }
        let o = cmd.output().expect("spawn camdl");
        assert!(
            o.status.success(),
            "generated draws under {scenario:?} must run; stderr:\n{}",
            stderr(&o)
        );
        run_leaves(&p.join(out).join("sims"))
    };

    let shadowed = simulate("with_scenario", Some("slow"));
    assert_eq!(
        shadowed.len(),
        1,
        "three draws the scenario overrides are one computation: {shadowed:?}"
    );
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(shadowed[0].join("run.json")).unwrap())
            .unwrap();
    let params_label = meta["levels"]
        .as_array()
        .and_then(|ls| ls.iter().find(|l| l["name"] == "params"))
        .and_then(|l| l["label"].as_str())
        .map(str::to_string);
    assert_eq!(
        params_label.as_deref(),
        Some("mu=0.01"),
        "the params label names the mu that ran; run.json levels: {}",
        meta["levels"]
    );

    let free = simulate("without_scenario", None);
    assert_eq!(free.len(), 3, "without the scenario the three draws are distinct: {free:?}");
}
