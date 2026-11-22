//! Transfers out of a compartment with an Erlang dwell, from model text to
//! trajectory (proposal docs/dev/proposals/2026-11-17-unified-staged-residence.md,
//! "Transfers and `add`", and its tests: stage-count invariance, ordering,
//! backends, reproducibility).
//!
//! A cohort starts in `E`, which progresses into `I` by an Erlang dwell.
//! Nothing else enters or leaves `I`, so at any recorded time the people
//! still in `E` or moved out of it by a transfer number `N0 - I`. A transfer
//! at day 5 moves some of `E` into `Q`; the tests check what it moved against
//! that total, at one stage (where `E` is a single cell and the transfer a
//! plain one) and at several (where it is a grouped transfer over the
//! stages), with the same model text.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn manifest() -> PathBuf {
    PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
}

fn binary() -> PathBuf {
    let bin = manifest().join("../../target/release/camdl");
    assert!(
        bin.exists(),
        "release camdl binary missing: {} — run `make build-rust` or `make test`",
        bin.display()
    );
    bin
}

fn camdl(dir: &Path) -> Command {
    let mut c = Command::new(binary());
    c.env_remove("CLICOLOR_FORCE")
        .env("NO_COLOR", "1")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_OUTPUT_DIR", dir.join("cas")).env("CAMDL_STORE", std::path::Path::new(&dir.join("cas")).join("camdl-store"))
        .current_dir(dir);
    c
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

/// The cohort model. `effects` is a whole `events { }` or `interventions { }`
/// block; `init` the body of `init { }`.
fn model(stages: u32, sigma: f64, init: &str, effects: &str) -> String {
    format!(
        r#"time_unit = 'days

compartments {{ E, I, Q, R }}

let sigma : rate = {sigma} 'per_day

transitions {{
  latency : E --> I via erlang(stages = {stages}, rate = sigma)
}}

init {{ {init} }}

{effects}

simulate {{ from = 0 'days  to = 8 'days }}
"#
    )
}

/// One row of a trajectory file, by column name.
struct Row(Vec<(String, f64)>);

impl Row {
    fn get(&self, name: &str) -> f64 {
        self.0
            .iter()
            .find(|(h, _)| h == name)
            .unwrap_or_else(|| {
                panic!("no column '{name}' in {:?}", self.0.iter().map(|r| &r.0).collect::<Vec<_>>())
            })
            .1
    }

    /// The stage cells of `E`: the single column `E`, or `E_s1`, `E_s2`, …
    fn e_stages(&self) -> Vec<f64> {
        self.0
            .iter()
            .filter(|(h, _)| h == "E" || h.starts_with("E_s"))
            .map(|(_, v)| *v)
            .collect()
    }
}

struct Run<'a> {
    tag: &'a str,
    src: String,
    backend: &'a str,
    seed: u64,
    enable: &'a [&'a str],
}

fn simulate(dir: &Path, run: &Run<'_>) -> Output {
    let src = dir.join(format!("{}.camdl", run.tag));
    std::fs::write(&src, &run.src).unwrap();
    let out = dir.join(format!("{}.tsv", run.tag));
    let _ = std::fs::remove_file(&out);
    let mut cmd = camdl(dir);
    // The integer backends step by a day. The ODE takes a short step, so the
    // RK4 error of the eight-stage chain is far below what the tests resolve,
    // and prints one row a day.
    let dt = if run.backend == "ode" { "0.01" } else { "1" };
    cmd.args(["simulate", src.to_str().unwrap(), "--backend", run.backend, "--dt", dt])
        .args(["--output-every", "1"])
        .args(["--seed", &run.seed.to_string(), "-o", out.to_str().unwrap()]);
    for name in run.enable {
        cmd.args(["--enable", name]);
    }
    cmd.output().expect("spawn simulate")
}

/// The trajectory row at time `t` of a run that must succeed.
fn row_at(dir: &Path, run: &Run<'_>, t: f64) -> Row {
    let o = simulate(dir, run);
    assert!(o.status.success(), "{}: simulate must succeed:\n{}", run.tag, stderr(&o));
    let tsv = std::fs::read_to_string(dir.join(format!("{}.tsv", run.tag))).unwrap();
    let mut lines = tsv.lines().filter(|l| !l.starts_with('#'));
    // The combined table opens with the coordinate columns (`replicate`,
    // `scenario`, `seed`); the trajectory's own columns start at `t`.
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let t_col = header.iter().position(|h| *h == "t").expect("a `t` column");
    assert_eq!(&header[..t_col], ["replicate", "scenario", "seed"], "{}: coordinate columns", run.tag);
    let header = &header[t_col..];
    for line in lines {
        let fields: Vec<f64> =
            line.split('\t').skip(t_col).map(|f| f.parse::<f64>().expect("number")).collect();
        if (fields[0] - t).abs() < 1e-9 {
            return Row(header.iter().map(|h| h.to_string()).zip(fields).collect());
        }
    }
    panic!("{}: no row at t = {t}", run.tag)
}

/// An `events { }` or `interventions { }` block with the one effect `pull`.
fn pull_block(block: &str, amount: &str) -> String {
    format!("{block} {{\n  pull : transfer({amount}, from = E, to = Q) at [5 'days]\n}}")
}

fn enabled(block: &str) -> &'static [&'static str] {
    if block == "interventions" { &["pull"] } else { &[] }
}

// ── Stage-count invariance ──────────────────────────────────────────────────

/// On the integer backends a transfer out of `E` moves one floor (or one
/// capped count) of the compartment's total, whatever the number of stages,
/// as an event and as an intervention, and leaves no stage negative.
#[test]
fn integer_backends_move_one_floor_of_the_compartment_total_at_any_stage_count() {
    let tmp = tempfile::tempdir().unwrap();
    type Expected = fn(f64) -> f64;
    let amounts: [(&str, &str, f64, Expected); 4] = [
        ("f04", "fraction = 0.4", 1000.0, |n| (n * 0.4).floor()),
        // A small cohort and a small fraction: a floor per stage would move
        // nobody from stages holding fewer than 15.
        ("fsmall", "fraction = 0.2 / 3", 60.0, |n| (n * (0.2 / 3.0)).floor()),
        ("c57", "count = 57", 1000.0, |n| n.min(57.0)),
        ("ccap", "count = 5000", 1000.0, |n| n),
    ];
    for backend in ["chain_binomial", "gillespie"] {
        for block in ["events", "interventions"] {
            for stages in [1u32, 3, 8] {
                for (name, amount, n0, expected) in amounts {
                    for seed in [1u64, 2] {
                        let tag = format!("inv_{backend}_{block}_{stages}_{name}_{seed}");
                        let run = Run {
                            tag: &tag,
                            src: model(stages, 0.2, &format!("E = {n0:.0}"), &pull_block(block, amount)),
                            backend,
                            seed,
                            enable: enabled(block),
                        };
                        let row = row_at(tmp.path(), &run, 5.0);
                        // Everyone not yet in I was in E when the transfer read it.
                        let total = n0 - row.get("I");
                        let moved = row.get("Q");
                        assert_eq!(
                            moved,
                            expected(total),
                            "{tag}: moved {moved} of a compartment total of {total}"
                        );
                        let left = row.e_stages();
                        assert_eq!(left.len(), stages as usize, "{tag}: one column per stage");
                        assert_eq!(left.iter().sum::<f64>(), total - moved, "{tag}: conservation");
                        assert!(left.iter().all(|&n| n >= 0.0), "{tag}: stages {left:?}");
                        // The check must not be vacuous: the cohort is spread
                        // over the stages when the transfer fires.
                        if stages == 3 && name == "f04" {
                            let before = row_at(tmp.path(), &run, 4.0).e_stages();
                            assert!(before.iter().all(|&n| n > 0.0), "{tag}: stages before {before:?}");
                        }
                    }
                }
            }
        }
    }
}

/// On the ODE nothing is rounded: a fraction of the total, or the count, at
/// any stage count.
#[test]
fn the_ode_moves_the_same_real_amount_at_any_stage_count() {
    let tmp = tempfile::tempdir().unwrap();
    // Large, so the whole numbers the trajectory prints resolve the amount to
    // one part in 1e9.
    let n0 = 1.0e9;
    type Expected = fn(f64) -> f64;
    let amounts: [(&str, &str, Expected); 3] = [
        ("f04", "fraction = 0.4", |n| n * 0.4),
        ("c", "count = 123456789", |n| n.min(123_456_789.0)),
        ("ccap", "count = 5000000000", |n| n),
    ];
    for block in ["events", "interventions"] {
        for stages in [1u32, 3, 8] {
            for (name, amount, expected) in amounts {
                let tag = format!("ode_{block}_{stages}_{name}");
                let run = Run {
                    tag: &tag,
                    src: model(stages, 0.2, &format!("E = {n0:.0}"), &pull_block(block, amount)),
                    backend: "ode",
                    seed: 1,
                    enable: enabled(block),
                };
                let row = row_at(tmp.path(), &run, 5.0);
                let total = n0 - row.get("I");
                let moved = row.get("Q");
                // Each printed compartment is rounded to a whole number.
                assert!(
                    (moved - expected(total)).abs() <= 2.0,
                    "{tag}: moved {moved}, expected {} of a total of {total}",
                    expected(total)
                );
                let left: f64 = row.e_stages().iter().sum();
                assert!((left - (total - moved)).abs() <= stages as f64 + 2.0, "{tag}: conservation");
            }
        }
    }
}

// ── Backends ────────────────────────────────────────────────────────────────

/// With no progression the state at the transfer is known exactly. Where the
/// real-valued amounts are whole numbers, chain-binomial, Gillespie and the
/// ODE give the same stage cells.
#[test]
fn the_backends_agree_where_the_real_valued_shares_are_whole() {
    let tmp = tempfile::tempdir().unwrap();
    let init = "stage(E, 1) = 300  stage(E, 2) = 500  stage(E, 3) = 200";
    for (name, amount, cells) in [
        ("frac", "fraction = 0.25", [225.0, 375.0, 150.0, 250.0]),
        ("count", "count = 100", [270.0, 450.0, 180.0, 100.0]),
    ] {
        for block in ["events", "interventions"] {
            for backend in ["chain_binomial", "gillespie", "ode"] {
                let tag = format!("agree_{name}_{block}_{backend}");
                let run = Run {
                    tag: &tag,
                    src: model(3, 0.0, init, &pull_block(block, amount)),
                    backend,
                    seed: 1,
                    enable: enabled(block),
                };
                let row = row_at(tmp.path(), &run, 5.0);
                let got = [row.get("E_s1"), row.get("E_s2"), row.get("E_s3"), row.get("Q")];
                assert_eq!(got, cells, "{tag}");
            }
        }
    }
}

/// The proposal's worked case through the whole pipeline: 5 people in stages
/// (3, 1, 1) at a fraction of 0.4 give up floor(5 · 0.4) = 2, as (1, 1, 0).
#[test]
fn five_people_in_three_stages_at_two_fifths() {
    let tmp = tempfile::tempdir().unwrap();
    for backend in ["chain_binomial", "gillespie"] {
        for block in ["events", "interventions"] {
            let tag = format!("worked_{backend}_{block}");
            let run = Run {
                tag: &tag,
                src: model(3, 0.0, "stage(E, 1) = 3  stage(E, 2) = 1  stage(E, 3) = 1", &pull_block(block, "fraction = 0.4")),
                backend,
                seed: 1,
                enable: enabled(block),
            };
            let row = row_at(tmp.path(), &run, 5.0);
            assert_eq!(
                [row.get("E_s1"), row.get("E_s2"), row.get("E_s3"), row.get("Q")],
                [2.0, 0.0, 1.0, 2.0],
                "{tag}"
            );
        }
    }
}

// ── Ordering ────────────────────────────────────────────────────────────────

/// As events at one instant, the inflow comes first whatever the declaration
/// order, and the transfer moves a share of the post-inflow total. As
/// interventions, the actions apply in declaration order.
#[test]
fn a_transfer_beside_an_add_at_the_same_time() {
    let tmp = tempfile::tempdir().unwrap();
    let pull = "  pull : transfer(fraction = 0.4, from = E, to = Q) at [5 'days]\n";
    let seed_e = "  seed : add(E, 100) at [5 'days]\n";
    for stages in [1u32, 3] {
        for backend in ["chain_binomial", "gillespie"] {
            for (order, body) in [("pull_first", format!("{pull}{seed_e}")), ("seed_first", format!("{seed_e}{pull}"))] {
                // Events: Q = floor(0.4 · (in E after the inflow)).
                let tag = format!("order_ev_{stages}_{backend}_{order}");
                let run = Run {
                    tag: &tag,
                    src: model(stages, 0.2, "E = 1000", &format!("events {{\n{body}}}")),
                    backend,
                    seed: 4,
                    enable: &[],
                };
                let row = row_at(tmp.path(), &run, 5.0);
                let before_inflow = 1000.0 - row.get("I");
                assert_eq!(row.get("Q"), ((before_inflow + 100.0) * 0.4).floor(), "{tag}");
                assert_eq!(
                    row.e_stages().iter().sum::<f64>(),
                    before_inflow + 100.0 - row.get("Q"),
                    "{tag}: conservation"
                );

                // Interventions: declaration order decides what the transfer reads.
                let tag = format!("order_iv_{stages}_{backend}_{order}");
                let run = Run {
                    tag: &tag,
                    src: model(stages, 0.2, "E = 1000", &format!("interventions {{\n{body}}}")),
                    backend,
                    seed: 4,
                    enable: &["pull", "seed"],
                };
                let row = row_at(tmp.path(), &run, 5.0);
                let before = 1000.0 - row.get("I");
                let read = if order == "pull_first" { before } else { before + 100.0 };
                assert_eq!(row.get("Q"), (read * 0.4).floor(), "{tag}");
            }
        }
    }
}

/// Two events draining `E` at one instant are refused on every backend. At one
/// stage they are two plain fraction transfers, refused because the fractions
/// sum above 1; at three they are two transfers out of one staged compartment,
/// refused whatever their fractions. A count beside another drain is refused
/// at either stage count.
#[test]
fn overdrawing_drains_of_a_staged_compartment_are_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let fractions = "events {\n  a : transfer(fraction = 0.6, from = E, to = Q) at [5 'days]\n  \
                     b : transfer(fraction = 0.6, from = E, to = R) at [5 'days]\n}";
    let with_count = "events {\n  a : transfer(count = 5, from = E, to = Q) at [5 'days]\n  \
                      b : transfer(fraction = 0.1, from = E, to = R) at [5 'days]\n}";
    for backend in ["chain_binomial", "gillespie", "ode"] {
        for (stages, cell, why) in [
            (1u32, "'E'", "`transfer` fractions from compartment 'E' sum to 1.2"),
            (3, "'E_s1'", "more than one `transfer` out of a staged compartment draws from 'E_s1'"),
        ] {
            let tag = format!("over_frac_{backend}_{stages}");
            let run = Run {
                tag: &tag,
                src: model(stages, 0.2, "E = 1000", fractions),
                backend,
                seed: 1,
                enable: &[],
            };
            let o = simulate(tmp.path(), &run);
            let err = stderr(&o);
            assert!(!o.status.success(), "{tag} must be refused:\n{err}");
            assert!(err.contains(why) && err.contains("events 'a', 'b'"), "{tag}: {err}");

            let tag = format!("over_count_{backend}_{stages}");
            let run = Run {
                tag: &tag,
                src: model(stages, 0.2, "E = 1000", with_count),
                backend,
                seed: 1,
                enable: &[],
            };
            let o = simulate(tmp.path(), &run);
            let err = stderr(&o);
            assert!(!o.status.success(), "{tag} must be refused:\n{err}");
            assert!(
                err.contains(&format!("more than one `transfer` from compartment {cell}"))
                    && err.contains("at least one moves a fixed count"),
                "{tag}: {err}"
            );
        }
    }
}

/// Two transfers out of one staged compartment at one instant are refused
/// before anything moves, even with fractions summing to exactly 1: each
/// divides its own total among the stages, and from stages (1, 1) two halves
/// would both take the individual in stage 1. The refusal names both events
/// and the shared cell, on every backend, and does not depend on the state.
#[test]
fn two_transfers_out_of_one_staged_compartment_at_one_instant_are_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let halves = "events {\n  a : transfer(fraction = 0.5, from = E, to = Q) at [5 'days]\n  \
                  b : transfer(fraction = 0.5, from = E, to = R) at [5 'days]\n}";
    for backend in ["chain_binomial", "gillespie", "ode"] {
        for (state, init) in [("one_each", "stage(E, 1) = 1  stage(E, 2) = 1"), ("many", "stage(E, 1) = 400  stage(E, 2) = 600")] {
            let tag = format!("shared_{backend}_{state}");
            let run = Run { tag: &tag, src: model(2, 0.0, init, halves), backend, seed: 1, enable: &[] };
            let o = simulate(tmp.path(), &run);
            let err = stderr(&o);
            assert!(!o.status.success(), "{tag} must be refused:\n{err}");
            assert!(
                err.contains("more than one `transfer` out of a staged compartment draws from 'E_s1' at t=5")
                    && err.contains("events 'a', 'b'"),
                "{tag}: {err}"
            );
            assert!(!err.contains("NegativeCount"), "{tag}: refused before any count goes negative:\n{err}");
        }
    }
    // The same two transfers at different times run.
    let apart = "events {\n  a : transfer(fraction = 0.5, from = E, to = Q) at [4 'days]\n  \
                 b : transfer(fraction = 0.5, from = E, to = R) at [5 'days]\n}";
    let run = Run {
        tag: "shared_apart",
        src: model(2, 0.0, "stage(E, 1) = 1  stage(E, 2) = 1", apart),
        backend: "chain_binomial",
        seed: 1,
        enable: &[],
    };
    let row = row_at(tmp.path(), &run, 5.0);
    assert_eq!([row.get("E_s1"), row.get("E_s2"), row.get("Q"), row.get("R")], [0.0, 1.0, 1.0, 0.0]);
}

// ── Trace ───────────────────────────────────────────────────────────────────

/// `CAMDL_TRACE_STEPS=1` reports a grouped transfer when it fires, naming the
/// cells it spans and the amount, as it reports every other action.
#[test]
fn trace_steps_reports_a_grouped_transfer() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    for (amount, want) in [
        ("fraction = 0.4", "EVENT 'pull' at t=5.0: grouped transfer E_s1 -> Q .. E_s3 -> Q (frac=0.40)"),
        ("count = 57", "EVENT 'pull' at t=5.0: grouped transfer E_s1 -> Q .. E_s3 -> Q (raw=57.00)"),
    ] {
        let src = dir.join("trace.camdl");
        std::fs::write(&src, model(3, 0.2, "E = 1000", &pull_block("events", amount))).unwrap();
        let o = camdl(dir)
            .env("CAMDL_TRACE_STEPS", "1")
            .env("CAMDL_NO_IR_CACHE", "1")
            .args(["simulate", "trace.camdl", "--backend", "chain_binomial", "--dt", "1"])
            .args(["--seed", "1", "-o", "trace.tsv"])
            .output()
            .expect("spawn simulate");
        let err = stderr(&o);
        assert!(o.status.success(), "{err}");
        assert_eq!(err.lines().filter(|l| l.contains(want)).count(), 1, "one line {want:?} in:\n{err}");
    }
}

// ── Reproducibility of inference ────────────────────────────────────────────

const OBSERVED: &str = r#"time_unit = 'days

compartments { E, I, Q }

parameters {
  sigma : rate in [0.01, 2.0]
  k     : positive in [0.1, 100]
}

transitions {
  latency : E --> I via erlang(stages = 3, rate = sigma)
}

init { E = 500 }

events {
  pull : transfer(fraction = 0.4, from = E, to = Q) at [3 'days]
}

observations {
  cases {
    columns       { time : time, cases : count }
    covers        = closing_at(time, 1 'days)
    projected     = incidence(latency)
    emit_schedule = every 1 'days
    cases ~ neg_binomial(mean = projected + 0.5, r = k)
  }
}

simulate { from = 0 'days  to = 6 'days }
"#;

const CASES: &str = "time\tcases\n1\t20\n2\t60\n3\t80\n4\t40\n5\t30\n6\t20\n";

/// A particle filter over a model with a grouped transfer gives the same
/// log-likelihood for the same seed, and a different one for another seed.
#[test]
fn a_particle_filter_with_a_grouped_transfer_is_reproducible() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    std::fs::write(dir.join("observed.camdl"), OBSERVED).unwrap();
    std::fs::write(dir.join("cases.tsv"), CASES).unwrap();
    let loglik = |seed: &str| -> String {
        let o = camdl(dir)
            .args(["pfilter", "observed.camdl", "--data", "cases.tsv", "--particles", "40"])
            .args(["--dt", "1", "--param", "sigma=0.3", "--param", "k=10", "--seed", seed])
            .output()
            .expect("spawn pfilter");
        assert!(o.status.success(), "pfilter must run:\n{}", stderr(&o));
        let out = String::from_utf8_lossy(&o.stdout).trim().to_string();
        assert!(out.parse::<f64>().is_ok_and(f64::is_finite), "a finite log-likelihood, got {out:?}");
        out
    };
    let first = loglik("7");
    assert_eq!(first, loglik("7"), "the same seed gives the same log-likelihood");
    assert_ne!(first, loglik("8"), "another seed gives another estimate");
}

/// A PGAS run over the same model writes the same trace, draws and
/// trajectories for the same seed.
#[test]
fn a_pgas_run_with_a_grouped_transfer_is_reproducible() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    std::fs::write(dir.join("observed.camdl"), OBSERVED).unwrap();
    std::fs::write(dir.join("cases.tsv"), CASES).unwrap();
    let run = |tag: &str| -> PathBuf {
        let out_dir = dir.join(format!("out_{tag}"));
        let fit = dir.join(format!("fit_{tag}.toml"));
        std::fs::write(
            &fit,
            format!(
                r#"store = "{out}"

[model]
camdl = "{model}"

[data.observations]
cases = "{data}"

[estimate]
sigma = {{ prior = {{ uniform = {{ lower = 0.01, upper = 2.0 }} }} }}

[fixed]
k = 10.0

[method]
algorithm = "pgas"
backend = "chain_binomial"
chains = 1
particles = 20
sweeps = 8
burn_in = 2
"#,
                out = out_dir.display(),
                model = dir.join("observed.camdl").display(),
                data = dir.join("cases.tsv").display(),
            ),
        )
        .unwrap();
        let o = camdl(dir)
            .args(["fit", "run"])
            .arg(&fit)
            .args(["--no-progress", "--seed", "5"])
            .output()
            .expect("spawn fit run");
        assert!(o.status.success(), "fit run must succeed:\n{}", stderr(&o));
        out_dir
    };
    /// The one file named `name` under `root`.
    fn find(root: &Path, name: &str) -> PathBuf {
        let mut stack = vec![root.to_path_buf()];
        let mut found = Vec::new();
        while let Some(d) = stack.pop() {
            for entry in std::fs::read_dir(&d).unwrap() {
                let p = entry.unwrap().path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.file_name().is_some_and(|f| f == name) {
                    found.push(p);
                }
            }
        }
        assert_eq!(found.len(), 1, "one {name} under {}: {found:?}", root.display());
        found.remove(0)
    }
    let (a, b) = (run("a"), run("b"));
    for name in ["trace.tsv", "draws.tsv", "trajectories.tsv"] {
        let (fa, fb) = (std::fs::read(find(&a, name)).unwrap(), std::fs::read(find(&b, name)).unwrap());
        assert!(!fa.is_empty(), "{name} is written");
        assert_eq!(fa, fb, "{name} is the same for the same seed");
    }
}

// ── An endpoint written with its stage ──────────────────────────────────────

/// `stage(I[pre], 3)` as a transfer source is that one location: the transfer
/// moves its fraction of stage 3 and leaves stages 1 and 2, and the other
/// stratum, as they were (proposal "Explicit stages"). The progression rate
/// is negligible, so the stages hold what `init` placed in them, which
/// `stage(...)` also writes.
#[test]
fn a_transfer_from_one_stage_moves_that_stage_only() {
    let tmp = tempfile::tempdir().unwrap();
    let src = r#"time_unit = 'days

dimensions { age = [pre, sch] }
compartments { I, Q, R }
stratify(by = age)

let gamma : rate = 0.000000001 'per_day

transitions {
  recovery[a in age] : I[a] --> R[a] via erlang(stages = 3, rate = gamma)
}

init {
  stage(I[pre], 1) = 100
  stage(I[pre], 2) = 200
  stage(I[pre], 3) = 400
  I[sch] = 50
}

interventions {
  isolate_late : transfer(fraction = 0.8, from = stage(I[pre], 3), to = Q[pre]) at [5 'days]
}

simulate { from = 0 'days  to = 8 'days }
"#;
    let run = |tag: &'static str, enable: &'static [&'static str]| Run {
        tag,
        src: src.to_string(),
        backend: "ode",
        seed: 1,
        enable,
    };
    let close = |tag: &str, row: &Row, name: &str, want: f64| {
        let got = row.get(name);
        assert!((got - want).abs() < 1e-4, "{tag}: {name} = {got}, expected {want}");
    };
    // Before the transfer, and in a run without it, the stages hold what
    // `init` placed: `stage(...)` in stages 1 to 3, the plain entry in stage 1.
    let before = row_at(tmp.path(), &run("stage_before", &["isolate_late"]), 4.0);
    let without = row_at(tmp.path(), &run("stage_without", &[]), 6.0);
    for (tag, row) in [("before", &before), ("without", &without)] {
        close(tag, row, "I_pre_s1", 100.0);
        close(tag, row, "I_pre_s2", 200.0);
        close(tag, row, "I_pre_s3", 400.0);
        close(tag, row, "I_sch_s1", 50.0);
        close(tag, row, "I_sch_s2", 0.0);
        close(tag, row, "Q_pre", 0.0);
    }
    let after = row_at(tmp.path(), &run("stage_after", &["isolate_late"]), 6.0);
    close("after", &after, "I_pre_s3", 80.0);
    close("after", &after, "Q_pre", 320.0);
    close("after", &after, "I_pre_s1", 100.0);
    close("after", &after, "I_pre_s2", 200.0);
    close("after", &after, "I_sch_s1", 50.0);
    close("after", &after, "Q_sch", 0.0);
}
