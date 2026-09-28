//! gh#572: a scenario that touches a parameter a user-authored coordinate
//! varies — a `[sweep]` / `[design.*]` point set, or a `--draws <file.tsv>` —
//! is refused before any cell runs (proposal 2026-06-27 §4).
//!
//! Precedence would let the scenario win: its `set` overrides, and its `scale`
//! rescales, every swept value, so the grid would not vary that parameter and
//! the store would hold leaves labelled with values that never ran. The
//! refusal covers every command that plans cells: `batch run`, `batch run
//! --dry-run`, `batch status`, and `simulate --draws <file>`. Generated draws
//! (`--draws uniform|prior|posterior`) are the negative control: a scenario
//! over generated draws is the counterfactual those runs exist for, and stays
//! allowed.
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

/// A scratch directory holding the model, a params file, and a batch manifest
/// running `scenario` over a three-point `mu` sweep.
struct Fixture {
    dir: tempfile::TempDir,
}

impl Fixture {
    fn new(scenario: &str) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        std::fs::write(p.join("m.camdl"), MODEL).unwrap();
        std::fs::write(p.join("p.toml"), "mu = 0.1\nnu = 0.1\n").unwrap();
        std::fs::write(
            p.join("batch.toml"),
            format!(
                r#"
[config]
model = "{model}"
params = "{params}"
output_dir = "{out}"
seeds = {{ n = 1 }}
backend = "ode"
parallel = 1

[[scenario]]
name = "{scenario}"

[sweep]
mu = [0.2, 0.3, 0.4]
"#,
                model = p.join("m.camdl").display(),
                params = p.join("p.toml").display(),
                out = p.join("out").display(),
            ),
        )
        .unwrap();
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

    /// `batch <sub> <manifest> <flags…>` — `batch run` / `batch status`.
    fn batch(&self, sub: &str, flags: &[&str]) -> Output {
        let manifest = self.path("batch.toml").to_string_lossy().into_owned();
        let mut args = vec!["batch", sub, manifest.as_str()];
        args.extend_from_slice(flags);
        self.camdl(&args)
    }

    /// Every directory under the output root holding a `run.json`.
    fn leaves(&self) -> Vec<PathBuf> {
        run_leaves(&self.path("out"))
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

/// `batch run`, `batch run --dry-run` and `batch status` all refuse the
/// manifest with the same diagnostic, and nothing is written to the store.
fn assert_every_batch_command_refuses(scenario: &str, lines: &[&str]) {
    let fx = Fixture::new(scenario);
    for (sub, flags) in [("run", &["--dry-run"][..]), ("status", &[][..]), ("run", &[][..])] {
        let what = format!("{scenario} / batch {sub} {}", flags.join(" "));
        assert_refused(&fx.batch(sub, flags), &what, lines);
    }
    assert!(fx.leaves().is_empty(), "{scenario}: a refused sweep must write no leaf");
}

#[test]
fn sweep_of_a_parameter_the_scenario_sets_is_refused() {
    assert_every_batch_command_refuses(
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
    assert_every_batch_command_refuses(
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
    assert_every_batch_command_refuses(
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
    // (cas_integration.rs `batch_sweep_records_sweep_point_in_run_json` covers
    // the same allowed shape on sir_basic with an ad-hoc scenario.)
    let fx = Fixture::new("pin_nu");
    let dry = fx.batch("run", &["--dry-run"]);
    assert!(dry.status.success(), "dry run must plan the grid; stderr:\n{}", stderr(&dry));
    let run = fx.batch("run", &[]);
    assert!(run.status.success(), "batch run must succeed; stderr:\n{}", stderr(&run));
    assert_eq!(fx.leaves().len(), 3, "three sweep points are three computations");
    let status = fx.batch("status", &[]);
    assert!(status.status.success(), "status must report; stderr:\n{}", stderr(&status));
    assert!(
        String::from_utf8_lossy(&status.stdout).contains("Completed:  3/3 leaves present"),
        "status counts the three leaves; stdout:\n{}",
        String::from_utf8_lossy(&status.stdout)
    );
}

#[test]
fn design_block_over_a_parameter_the_scenario_sets_is_refused() {
    // A `[design.*]` block's points run as a sweep, so the same refusal applies,
    // before the block's `parameter_points.tsv` is written.
    let fx = Fixture::new("pin_mu");
    std::fs::write(
        fx.path("design.toml"),
        format!(
            r#"
[config]
model = "{model}"
params = "{params}"
output_dir = "{out}"
seeds = {{ n = 1 }}
backend = "ode"
parallel = 1

[[scenario]]
name = "pin_mu"

[design.sens]
method = "random"
n = 3
parameters.mu = {{ range = {{ min = 0.05, max = 0.5 }} }}
"#,
            model = fx.path("m.camdl").display(),
            params = fx.path("p.toml").display(),
            out = fx.path("out").display(),
        ),
    )
    .unwrap();
    let manifest = fx.path("design.toml").to_string_lossy().into_owned();
    for extra in [&["--dry-run"][..], &[][..]] {
        let mut args = vec!["batch", "run", manifest.as_str()];
        args.extend_from_slice(extra);
        assert_refused(
            &fx.camdl(&args),
            "design × pin_mu",
            &["error: parameter `mu` is controlled by both the sweep and scenario `pin_mu`"],
        );
    }
    assert!(!fx.path("out").exists(), "a refused design writes nothing under the output root");
}

#[test]
fn draws_file_column_the_scenario_sets_is_refused_with_the_shared_diagnostic() {
    let fx = Fixture::new("pin_mu");
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
    let fx = Fixture::new("pin_mu");
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
    let fx = Fixture::new("half_mu");
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
/// ran" invariant (gh#583) is still reachable end to end: the batch sweep
/// that used to exercise it is now refused by this issue's guard.
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
