//! A time function evaluated outside the range on which it is defined is an
//! error unless the run is given `--allow-forcing-extrapolation` (or, in a fit
//! config, `[config] allow_forcing_extrapolation = true`). Under the override
//! the end value is held exactly as before, one warning per function names its
//! range and the span held, and the override is recorded in `run.json`.
//!
//! One model per kind that has a range: an `interpolated` forcing read
//! linearly, as a cubic spline and piecewise-constant (`method = linear |
//! spline | constant`), and a `bspline` over a window. Each forcing's data, or
//! window, ends at t = 50 while the run goes to t = 100.
//!
//! Spec: docs/dev/temporal-architecture.md §4.6 (D44, D48); proposal
//! docs/dev/proposals/2026-11-15-temporal-architecture-migration.md Stage 0(f).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn bin() -> PathBuf {
    let p = repo().join("rust/target/release/camdl");
    assert!(p.exists(), "release camdl binary missing: {} — run `make build`", p.display());
    p
}

/// The forcing block of each kind; `end` is the last data point or the
/// window end.
fn forcing(kind: &str, end: u32) -> String {
    match kind {
        "linear" | "spline" | "constant" => format!(
            "  mult : interpolated 'ratio {{\n    data = \"mult.tsv\"\n    time_col = \"time\"\n    \
             value_col = \"mult\"\n    method = {kind}\n  }}"
        ),
        "bspline" => format!(
            "  mult : bspline 'ratio {{\n    window = [0 'days, {end} 'days]\n    n_basis = 4\n    \
             coefs = [1.0, 0.8, 0.6, 0.5]\n  }}"
        ),
        other => panic!("no kind {other}"),
    }
}

const KINDS: [&str; 4] = ["linear", "spline", "constant", "bspline"];

/// An SIR model whose infection rate is scaled by `mult`, observed weekly,
/// run from t = 0 to t = 100. For the data-table kinds, `mult.tsv` holds one
/// row per `rows` entry.
fn model(dir: &Path, kind: &str, rows: &[(u32, f64)]) -> PathBuf {
    let mut tsv = String::from("time\tmult\n");
    for (t, v) in rows {
        tsv.push_str(&format!("{t}\t{v}\n"));
    }
    std::fs::write(dir.join("mult.tsv"), tsv).unwrap();
    let end = rows.last().map_or(50, |r| r.0);
    let p = dir.join(format!("{kind}.camdl"));
    std::fs::write(
        &p,
        format!(
            r#"time_unit = 'days

compartments {{ S, I, R }}

parameters {{
  beta  : rate in [0.05, 2.0]
  gamma : rate in [0.01, 1.0]
  rho   : probability in [0.05, 0.95]
}}

let N = S + I + R

forcing {{
{forcing}
}}

transitions {{
  infection : S --> I @ mult * beta * S * (I / N)
  recovery  : I --> R @ gamma * I
}}

init {{ S = 990  I = 10 }}

observations {{
  cases {{
    columns       {{ time : time, cases : count }}
    covers        = closing_at(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    cases         ~ poisson(rate = rho * projected)
  }}
}}

simulate {{ from = 0 'days  to = 100 'days }}
"#,
            forcing = forcing(kind, end)
        ),
    )
    .unwrap();
    p
}

/// Data from t = 0 to t = 50, weekly-ish, ending on a value other than its
/// neighbours so a hold is visible.
const ROWS_TO_50: [(u32, f64); 6] = [(0, 1.0), (10, 0.9), (20, 0.7), (30, 0.8), (40, 0.6), (50, 0.5)];

fn camdl(dir: &Path, args: &[&str]) -> Output {
    Command::new(bin())
        .env_remove("CLICOLOR_FORCE")
        .args(args)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_IR_CACHE_DIR", dir.join("irc"))
        .env("CAMDL_OUTPUT_DIR", dir.join("cas")).env("CAMDL_STORE", std::path::Path::new(&dir.join("cas")).join("camdl-store"))
        .output()
        .expect("spawn camdl")
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

/// `camdl simulate` of `model`, its trajectory mirrored to `traj.tsv`.
fn simulate(dir: &Path, model: &Path, extra: &[&str]) -> Output {
    std::fs::write(dir.join("params.toml"), "beta = 0.3\ngamma = 0.1\nrho = 0.6\n").unwrap();
    let traj = dir.join("traj.tsv");
    let params = dir.join("params.toml");
    let mut args = vec![
        "simulate",
        model.to_str().unwrap(),
        "--backend",
        "chain_binomial",
        "--seed",
        "1",
        "--dt",
        "1",
        "--params",
        params.to_str().unwrap(),
        "-o",
        traj.to_str().unwrap(),
    ];
    args.extend_from_slice(extra);
    camdl(dir, &args)
}

/// A tiny PGAS `fit run` of `model` on weekly counts over days 7–70, past the
/// forcing's range, with `config_extra` appended to `[config]`.
fn fit_run(dir: &Path, model: &Path, config_extra: &str) -> Output {
    let toml = fit_toml(dir, model, config_extra);
    camdl(dir, &["fit", "run", toml.to_str().unwrap()])
}

/// The fit config [`fit_run`] runs, written to `fit.toml` in `dir`.
fn fit_toml(dir: &Path, model: &Path, config_extra: &str) -> PathBuf {
    let data = dir.join("cases.tsv");
    let mut body = String::from("time\tcases\n");
    for w in 1..=10 {
        body.push_str(&format!("{}\t{}\n", 7 * w, 10 + 3 * w));
    }
    std::fs::write(&data, body).unwrap();
    let toml = dir.join("fit.toml");
    std::fs::write(
        &toml,
        format!(
            "output_dir = \"{out}\"\n[model]\ncamdl = \"{model}\"\n[data.observations]\ncases = \"{data}\"\n\
             [config]\ndt = 1.0\n{config_extra}[estimate]\n\
             gamma = {{ bounds = [0.01, 1.0], start = 0.1, prior = {{ uniform = {{}} }} }}\n\
             [fixed]\nbeta = 0.3\nrho = 0.6\n\
             [method]\nalgorithm = \"pgas\"\nbackend = \"chain_binomial\"\nchains = 1\nparticles = 8\n\
             sweeps = 4\nburn_in = 2\n",
            out = dir.join("results").display(),
            model = model.display(),
            data = data.display(),
        ),
    )
    .unwrap();
    toml
}

/// The one line of the warning a held forcing gets, by its code.
const WARNING: &str = "forcing-outside-range";

fn warnings(stderr: &str) -> usize {
    stderr.matches("forcing `mult` is defined on").count()
}

/// Each kind, past its range under `simulate`, is refused: a non-zero exit
/// naming the forcing, its range, the side it was evaluated on and the
/// override.
#[test]
fn simulate_refuses_each_kind_past_its_range() {
    for kind in KINDS {
        let tmp = tempfile::tempdir().unwrap();
        let m = model(tmp.path(), kind, &ROWS_TO_50);
        let o = simulate(tmp.path(), &m, &[]);
        let err = stderr(&o);
        assert!(!o.status.success(), "{kind}: a run past the forcing's range must be refused:\n{err}");
        assert!(err.contains("forcing `mult`"), "{kind}: names the forcing:\n{err}");
        assert!(err.contains("t = 0 to t = 50"), "{kind}: names the range:\n{err}");
        assert!(err.contains("after its last data point") || err.contains("after the end of its window"),
            "{kind}: names the side it was evaluated on:\n{err}");
        // No time outside the range is named: which one a multi-threaded run
        // meets first depends on its threads.
        assert!(!err.contains("t = 51"), "{kind}: names no time outside the range:\n{err}");
        assert!(err.contains("--allow-forcing-extrapolation"), "{kind}: names the override:\n{err}");
        assert!(!tmp.path().join("traj.tsv").exists(), "{kind}: a refused run writes no trajectory");
    }
}

/// The same under `fit run`, and `[config] allow_forcing_extrapolation = true`
/// lets the fit run, with one warning.
#[test]
fn fit_run_refuses_each_kind_past_its_range() {
    for kind in KINDS {
        let tmp = tempfile::tempdir().unwrap();
        let m = model(tmp.path(), kind, &ROWS_TO_50);
        let o = fit_run(tmp.path(), &m, "");
        let err = stderr(&o);
        assert!(!o.status.success(), "{kind}: fit run past the forcing's range must be refused:\n{err}");
        assert!(err.contains("forcing `mult`") && err.contains("--allow-forcing-extrapolation"), "{kind}:\n{err}");

        let tmp = tempfile::tempdir().unwrap();
        let m = model(tmp.path(), kind, &ROWS_TO_50);
        let o = fit_run(tmp.path(), &m, "allow_forcing_extrapolation = true\n");
        let err = stderr(&o);
        assert!(o.status.success(), "{kind}: the config override lets the fit run:\n{err}");
        assert_eq!(warnings(&err), 1, "{kind}: one warning for the one forcing:\n{err}");
    }
}

/// What a `run.json` records of a run that held `mult` past its range at
/// t = 50 under the override.
fn held_mult_past_50() -> serde_json::Value {
    serde_json::json!({
        "allowed": true,
        "held": [{ "forcing": "mult", "defined_on": [0.0, 50.0], "side": "after", "held_value_at": 50.0 }],
    })
}

/// A fit under `[config] allow_forcing_extrapolation = true` records the
/// override and the one forcing it held in its method leaf's `run.json`, as a
/// simulate under the flag does in its leaf's: the record is the command's,
/// whichever way the override was given.
#[test]
fn the_override_records_the_forcing_it_held() {
    let tmp = tempfile::tempdir().unwrap();
    let m = model(tmp.path(), "linear", &ROWS_TO_50);
    let o = fit_run(tmp.path(), &m, "allow_forcing_extrapolation = true\n");
    assert!(o.status.success(), "{}", stderr(&o));
    let records = statuses(&tmp.path().join("results"));
    let method: Vec<_> = records.iter().filter(|(_, v)| v["kind"] != "fit").collect();
    assert!(!method.is_empty(), "a method leaf was written: {records:?}");
    for (path, v) in method {
        assert_eq!(v["inputs"]["forcing_extrapolation"], held_mult_past_50(), "{}:\n{v:#}", path.display());
    }

    let o = simulate(tmp.path(), &m, &["--allow-forcing-extrapolation"]);
    assert!(o.status.success(), "{}", stderr(&o));
    let records = statuses(&tmp.path().join("cas"));
    assert!(!records.is_empty(), "the simulate leaf was written");
    for (path, v) in records {
        assert_eq!(v["inputs"]["forcing_extrapolation"], held_mult_past_50(), "{}:\n{v:#}", path.display());
    }
}

/// `pfilter` and `profile` given `--fit` read the config's `[config]
/// allow_forcing_extrapolation`, which `fit run` applied to the fit: under it
/// alone each runs past the forcing's range, records what it held, and is the
/// same computation as under the flag, so the second run adds no run id
/// (gh#1235).
#[test]
fn fit_aware_commands_read_the_configs_override() {
    let tmp = tempfile::tempdir().unwrap();
    let m = model(tmp.path(), "linear", &ROWS_TO_50);
    let toml = fit_toml(tmp.path(), &m, "allow_forcing_extrapolation = true\n");
    std::fs::write(tmp.path().join("params.toml"), "beta = 0.3\ngamma = 0.1\nrho = 0.6\n").unwrap();
    let params = tmp.path().join("params.toml");
    let (m, toml, params) = (m.to_str().unwrap(), toml.to_str().unwrap(), params.to_str().unwrap());
    let prof = tmp.path().join("prof");
    let prof = prof.to_str().unwrap();
    let commands: [(&str, Vec<&str>); 2] = [
        ("pfilter", vec!["pfilter", m, "--fit", toml, "--params", params, "--particles", "50", "--seed", "1", "--dt", "1"]),
        ("profile", vec![
            "profile", m, "--fit", toml, "--sweep", "gamma=0.1,0.2", "--particles", "8", "--iterations", "2",
            "--n-starts", "1", "--seed", "1", "-o", prof,
        ]),
    ];
    for (name, args) in commands {
        let before = run_ids(tmp.path());
        let o = camdl(tmp.path(), &args);
        let err = stderr(&o);
        assert!(o.status.success(), "{name}: the config's override lets it run:\n{err}");
        assert_eq!(warnings(&err), 1, "{name}: one warning for the one forcing:\n{err}");
        let after = run_ids(tmp.path());
        assert!(after.len() > before.len(), "{name}: stored a run");
        let new: Vec<_> = statuses(tmp.path())
            .into_iter()
            .filter(|(_, v)| v["run_id"].as_str().is_some_and(|id| !before.contains(id)))
            .filter(|(_, v)| v["status"] == "completed" && v["kind"] != "profile_invocation")
            .collect();
        assert!(!new.is_empty(), "{name}: a completed leaf");
        for (path, v) in &new {
            assert_eq!(v["inputs"]["forcing_extrapolation"], held_mult_past_50(), "{name}: {}:\n{v:#}", path.display());
        }
        let mut flagged = args.clone();
        flagged.push("--allow-forcing-extrapolation");
        let o = camdl(tmp.path(), &flagged);
        assert!(o.status.success(), "{name}: under the flag too:\n{}", stderr(&o));
        assert_eq!(run_ids(tmp.path()), after, "{name}: the config's override and the flag are one computation");
    }
}

/// A fit made with `--allow-forcing-extrapolation` on the command line, not in
/// its config, stores the config it ran under: the fit folder keeps the file
/// byte for byte as `fit.toml.original`, comments included, and its
/// `fit.resolved.toml` records `[config] allow_forcing_extrapolation = true`,
/// so every reader of it — `fit predict` here — runs under the override
/// without being given the flag again (gh#1268).
#[test]
fn a_fit_made_under_the_flag_stores_the_override_in_its_config() {
    let tmp = tempfile::tempdir().unwrap();
    let m = model(tmp.path(), "linear", &ROWS_TO_50);
    let toml = fit_toml(tmp.path(), &m, "");
    let written = format!("# the user's own comment\n{}", std::fs::read_to_string(&toml).unwrap());
    std::fs::write(&toml, &written).unwrap();
    let o = camdl(tmp.path(), &["fit", "run", toml.to_str().unwrap(), "--allow-forcing-extrapolation"]);
    assert!(o.status.success(), "the flag lets the fit run:\n{}", stderr(&o));
    let leaf = statuses(&tmp.path().join("results"))
        .into_iter()
        .find(|(_, v)| v["kind"] == "fit_method")
        .map(|(p, _)| p.parent().unwrap().to_path_buf())
        .expect("the fit's method leaf");
    let segment = leaf.parent().unwrap().parent().unwrap();
    let archived = std::fs::read_to_string(segment.join("fit.toml.original")).unwrap();
    assert_eq!(archived, written, "fit.toml.original is the user's file, byte for byte");
    let resolved = std::fs::read_to_string(segment.join("fit.resolved.toml")).unwrap();
    assert!(resolved.starts_with("# fit.resolved.toml:"), "the resolved config says what it is:\n{resolved}");
    let parsed: toml::Table = resolved.parse().unwrap_or_else(|e| panic!("the resolved config parses: {e}\n{resolved}"));
    let o = camdl(tmp.path(), &["fit", "predict", leaf.to_str().unwrap(), "--mode", "free_forward"]);
    let err = stderr(&o);
    assert!(!err.contains("forcing `mult` was evaluated"), "predict reads the stored override:\n{err}");
    assert!(o.status.success(), "{err}");
    assert_eq!(
        parsed["config"].get("allow_forcing_extrapolation").and_then(|v| v.as_bool()),
        Some(true),
        "the resolved config records the override the fit ran under:\n{resolved}"
    );
}

/// The same fit, named by its `fit.toml`: the file alone does not hold the
/// override, so the flag-made run is not the run `fit run fit.toml` makes,
/// and a `fit.toml` handle names only that run (gh#896). `fit predict` and
/// `compare` refuse it rather than replay the flag-made fit under a config
/// without the override, and the refusal names the flag-made run's leaf
/// directory so the reader can name it instead (gh#1268).
#[test]
fn a_fit_toml_handle_does_not_stand_for_a_flag_made_fit() {
    let tmp = tempfile::tempdir().unwrap();
    let m = model(tmp.path(), "linear", &ROWS_TO_50);
    let toml = fit_toml(tmp.path(), &m, "");
    let toml = toml.to_str().unwrap();
    let o = camdl(tmp.path(), &["fit", "run", toml, "--allow-forcing-extrapolation"]);
    assert!(o.status.success(), "the flag lets the fit run:\n{}", stderr(&o));
    let leaf = statuses(&tmp.path().join("results"))
        .into_iter()
        .find(|(_, v)| v["kind"] == "fit_method")
        .map(|(p, _)| p.parent().unwrap().to_path_buf())
        .expect("the fit's method leaf");
    let leaf_name = leaf.file_name().unwrap().to_string_lossy().into_owned();
    for args in [
        vec!["fit", "predict", toml, "--mode", "free_forward"],
        vec!["compare", toml, leaf.to_str().unwrap()],
    ] {
        let o = camdl(tmp.path(), &args);
        let err = stderr(&o);
        assert!(!o.status.success(), "{args:?}: the file alone names no run here:\n{err}");
        assert!(!err.contains("forcing `mult` was evaluated"), "{args:?}: not replayed without the override:\n{err}");
        assert!(err.contains("command-line options"), "{args:?}: says why the stored run is not the file's:\n{err}");
        assert!(err.contains(&leaf_name), "{args:?}: names the flag-made run's leaf:\n{err}");
    }
}

/// The commands that read a fit's problem but none of its results (`pfilter`,
/// `profile`), given a fit made with `--allow-forcing-extrapolation` on the
/// command line: named by its leaf, the fit's stored config holds the
/// override, so the command runs past the forcing's range and records what it
/// held (gh#1235, gh#1268). For `pfilter` it is the same
/// computation as the file under the flag; `profile` keys the config's text as read, and the
/// stored config is the file with the override written in, so the two are
/// stored apart (a conservative re-key, not a shared leaf). Named by its
/// `fit.toml`, the problem is the file's, which does not hold the override
/// (gh#896), so the same command is refused at the forcing rather than run as
/// the flag-made fit.
#[test]
fn a_flag_made_fit_is_read_under_its_stored_override_through_its_leaf() {
    let tmp = tempfile::tempdir().unwrap();
    let m = model(tmp.path(), "linear", &ROWS_TO_50);
    let toml = fit_toml(tmp.path(), &m, "");
    std::fs::write(tmp.path().join("params.toml"), "beta = 0.3\ngamma = 0.1\nrho = 0.6\n").unwrap();
    let params = tmp.path().join("params.toml");
    let o = camdl(tmp.path(), &["fit", "run", toml.to_str().unwrap(), "--allow-forcing-extrapolation"]);
    assert!(o.status.success(), "the flag lets the fit run:\n{}", stderr(&o));
    let leaf = statuses(&tmp.path().join("results"))
        .into_iter()
        .find(|(_, v)| v["kind"] == "fit_method")
        .map(|(p, _)| p.parent().unwrap().to_path_buf())
        .expect("the fit's method leaf");
    let (m, toml, params, leaf) = (m.to_str().unwrap(), toml.to_str().unwrap(), params.to_str().unwrap(), leaf.to_str().unwrap());
    let prof = tmp.path().join("prof");
    let prof = prof.to_str().unwrap();
    // Each command, and whether the stored override and the flag on the file
    // are one stored computation.
    type Args<'a> = Box<dyn Fn(&str) -> Vec<String> + 'a>;
    let commands: [(&str, Args, bool); 2] = [
        (
            "pfilter",
            Box::new(|fit: &str| {
                ["pfilter", m, "--fit", fit, "--params", params, "--particles", "50", "--seed", "1", "--dt", "1"]
                    .map(String::from)
                    .to_vec()
            }),
            true,
        ),
        (
            "profile",
            Box::new(|fit: &str| {
                [
                    "profile", m, "--fit", fit, "--sweep", "gamma=0.1,0.2", "--particles", "8", "--iterations", "2",
                    "--n-starts", "1", "--seed", "1", "-o", prof,
                ]
                .map(String::from)
                .to_vec()
            }),
            false,
        ),
    ];
    for (name, args, one_computation) in commands {
        let before = run_ids(tmp.path());
        let by_leaf = args(leaf);
        let o = camdl(tmp.path(), &by_leaf.iter().map(String::as_str).collect::<Vec<_>>());
        let err = stderr(&o);
        assert!(o.status.success(), "{name} --fit <leaf>: the stored override lets it run:\n{err}");
        assert_eq!(warnings(&err), 1, "{name}: one warning for the one forcing:\n{err}");
        let after = run_ids(tmp.path());
        let new: Vec<_> = statuses(tmp.path())
            .into_iter()
            .filter(|(_, v)| v["run_id"].as_str().is_some_and(|id| !before.contains(id)))
            .filter(|(_, v)| v["status"] == "completed" && v["kind"] != "profile_invocation" && v["kind"] != "fit_method")
            .collect();
        assert!(!new.is_empty(), "{name}: a completed leaf");
        for (path, v) in &new {
            assert_eq!(v["inputs"]["forcing_extrapolation"], held_mult_past_50(), "{name}: {}:\n{v:#}", path.display());
        }
        let mut flagged = args(toml);
        flagged.push("--allow-forcing-extrapolation".into());
        let o = camdl(tmp.path(), &flagged.iter().map(String::as_str).collect::<Vec<_>>());
        assert!(o.status.success(), "{name} --fit fit.toml under the flag:\n{}", stderr(&o));
        if one_computation {
            assert_eq!(run_ids(tmp.path()), after, "{name}: the stored override and the flag are one computation");
        }
        let o = camdl(tmp.path(), &args(toml).iter().map(String::as_str).collect::<Vec<_>>());
        let err = stderr(&o);
        assert!(!o.status.success(), "{name} --fit fit.toml: the file holds no override:\n{err}");
        assert!(err.contains("forcing `mult` was evaluated"), "{name}: refused at the forcing:\n{err}");
    }
}

/// `compare` scores a chain-binomial fit with a `camdl pfilter` child at its
/// posterior draws. A fit made under `[config] allow_forcing_extrapolation`
/// is scored under it, so the child holds the forcing rather than refusing
/// it (gh#1235).
#[test]
fn compare_scores_a_fit_under_its_configs_override() {
    let tmp = tempfile::tempdir().unwrap();
    let m = model(tmp.path(), "linear", &ROWS_TO_50);
    let o = fit_run(tmp.path(), &m, "allow_forcing_extrapolation = true\n");
    assert!(o.status.success(), "{}", stderr(&o));
    let leaf = statuses(&tmp.path().join("results"))
        .into_iter()
        .find(|(p, v)| v["kind"] != "fit" && p.to_string_lossy().contains("pgas-"))
        .map(|(p, _)| p.parent().unwrap().to_string_lossy().into_owned())
        .expect("the fit's method leaf");
    let o = camdl(tmp.path(), &["compare", &leaf, &leaf]);
    let err = stderr(&o);
    assert!(!err.contains("camdl does not extend a forcing past its range"), "the child must not refuse:\n{err}");
    assert!(o.status.success(), "{err}");
}

/// The method leaf of the one fit under `results` in `dir` whose method
/// directory starts with `method`.
fn method_leaf(dir: &Path, method: &str) -> String {
    statuses(&dir.join("results"))
        .into_iter()
        .find(|(p, v)| v["kind"] != "fit" && p.to_string_lossy().contains(&format!("{method}-")))
        .map(|(p, _)| p.parent().unwrap().to_string_lossy().into_owned())
        .expect("the fit's method leaf")
}

/// `compare` scores an ODE fit in-process, by the fit's own ODE likelihood
/// rather than a `camdl pfilter` child. A fit made under `[config]
/// allow_forcing_extrapolation` is scored under it there too: the scorer holds
/// the forcing rather than refusing it (gh#1235).
#[test]
fn compare_scores_an_ode_fit_under_its_configs_override() {
    let tmp = tempfile::tempdir().unwrap();
    let m = model(tmp.path(), "linear", &ROWS_TO_50);
    let toml = fit_toml(tmp.path(), &m, "allow_forcing_extrapolation = true\n");
    let text = std::fs::read_to_string(&toml).unwrap().replace(
        "[method]\nalgorithm = \"pgas\"\nbackend = \"chain_binomial\"\nchains = 1\nparticles = 8\n\
         sweeps = 4\nburn_in = 2\n",
        "[method]\nalgorithm = \"mh\"\nbackend = \"ode\"\nchains = 1\niterations = 40\nburn_in = 10\n\
         starts = \"single\"\n",
    );
    assert!(text.contains("algorithm = \"mh\""), "the config is the ODE fit's:\n{text}");
    std::fs::write(&toml, text).unwrap();
    let o = camdl(tmp.path(), &["fit", "run", toml.to_str().unwrap()]);
    assert!(o.status.success(), "{}", stderr(&o));
    let leaf = method_leaf(tmp.path(), "mh");
    let o = camdl(tmp.path(), &["compare", &leaf, &leaf]);
    let err = stderr(&o);
    assert!(!err.contains("camdl does not extend a forcing past its range"), "the scorer must not refuse:\n{err}");
    assert!(o.status.success(), "{err}");
}

/// A fit of the linear-forcing model in `dir` held out after day 49, with
/// `config_extra` appended to `[config]`, by the method `fit_toml` writes
/// (PGAS on the chain-binomial backend) or, with `ode`, MH on the ODE
/// backend; returns its method leaf. The held-out score reaches day 70, past
/// the forcing's range.
fn held_out_fit(dir: &Path, config_extra: &str, ode: bool) -> String {
    let m = model(dir, "linear", &ROWS_TO_50);
    let toml = fit_toml(dir, &m, config_extra);
    let mut text = std::fs::read_to_string(&toml)
        .unwrap()
        .replace("[data.observations]", "[data]\nholdout_after = 49\n[data.observations]");
    if ode {
        text = text.replace(
            "[method]\nalgorithm = \"pgas\"\nbackend = \"chain_binomial\"\nchains = 1\nparticles = 8\n\
             sweeps = 4\nburn_in = 2\n",
            "[method]\nalgorithm = \"mh\"\nbackend = \"ode\"\nchains = 1\niterations = 40\nburn_in = 10\n\
             starts = \"single\"\n",
        );
    }
    assert!(text.contains("holdout_after") && (!ode || text.contains("algorithm = \"mh\"")), "{text}");
    std::fs::write(&toml, text).unwrap();
    let o = camdl(dir, &["fit", "run", toml.to_str().unwrap()]);
    assert!(o.status.success(), "{}", stderr(&o));
    method_leaf(dir, if ode { "mh" } else { "pgas" })
}

/// The lines of `compare`'s report that say a row was scored with `mult`
/// held outside its range.
fn held_lines(o: &Output) -> Vec<String> {
    let all = format!("{}{}", String::from_utf8_lossy(&o.stdout), stderr(o));
    all.lines().filter(|l| l.contains("scored with forcing `mult` held outside its range")).map(str::to_string).collect()
}

/// `compare` reports, per row, every forcing held while that row was
/// scored. A chain-binomial fit is scored by a `camdl pfilter` child, which
/// records what it held in the trace it writes; a fit made under the
/// override, held out after day 49, holds the forcing only when its held-out
/// days are scored, so its fit never said so (gh#1235).
#[test]
fn compare_reports_a_hold_made_while_scoring_each_row() {
    let tmp = tempfile::tempdir().unwrap();
    let leaf = held_out_fit(tmp.path(), "allow_forcing_extrapolation = true\n", false);
    let o = camdl(tmp.path(), &["compare", &leaf, &leaf]);
    assert!(o.status.success(), "{}", stderr(&o));
    assert_eq!(held_lines(&o).len(), 2, "one line per row:\n{}\n{}", String::from_utf8_lossy(&o.stdout), stderr(&o));
}

/// A fit whose method is `pfilter` writes its prequential trace in its
/// leaf, and `compare` reads that trace as it is. Made under the override on
/// data past the forcing's range, the filter held the forcing, and the trace
/// records it, so `compare` reports it on each row (gh#1235).
#[test]
fn compare_reports_a_hold_recorded_by_a_pfilter_method_fit() {
    let tmp = tempfile::tempdir().unwrap();
    let m = model(tmp.path(), "linear", &ROWS_TO_50);
    let toml = fit_toml(tmp.path(), &m, "allow_forcing_extrapolation = true\n");
    let text = std::fs::read_to_string(&toml).unwrap().replace(
        "[method]\nalgorithm = \"pgas\"\nbackend = \"chain_binomial\"\nchains = 1\nparticles = 8\n\
         sweeps = 4\nburn_in = 2\n",
        "[method]\nalgorithm = \"pfilter\"\nbackend = \"chain_binomial\"\nparticles = 50\nreplicates = 1\n",
    );
    assert!(text.contains("algorithm = \"pfilter\""), "{text}");
    std::fs::write(&toml, text).unwrap();
    let o = camdl(tmp.path(), &["fit", "run", toml.to_str().unwrap()]);
    assert!(o.status.success(), "{}", stderr(&o));
    let leaf = method_leaf(tmp.path(), "pfilter");
    let o = camdl(tmp.path(), &["compare", &leaf, &leaf]);
    assert!(o.status.success(), "{}", stderr(&o));
    assert_eq!(held_lines(&o).len(), 2, "one line per row:\n{}\n{}", String::from_utf8_lossy(&o.stdout), stderr(&o));
}

/// A fit made without the override, scored under compare's
/// `--allow-forcing-extrapolation`: compare says the flag turned the setting
/// on for it, and reports the hold on its rows.
#[test]
fn compare_says_when_its_flag_scores_a_fit_under_the_override() {
    let tmp = tempfile::tempdir().unwrap();
    let leaf = held_out_fit(tmp.path(), "", false);
    let o = camdl(tmp.path(), &["compare", &leaf, &leaf, "--allow-forcing-extrapolation"]);
    let err = stderr(&o);
    assert!(o.status.success(), "{err}");
    assert!(err.contains("was fitted without --allow-forcing-extrapolation; compare scores it under it, as given"),
        "{err}");
    assert_eq!(held_lines(&o).len(), 2, "{}\n{err}", String::from_utf8_lossy(&o.stdout));
    // `--format json` records it in each row's trace derivation.
    let o = camdl(tmp.path(), &["compare", &leaf, &leaf, "--allow-forcing-extrapolation", "--format", "json"]);
    assert!(o.status.success(), "{}", stderr(&o));
    let v: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
    let text = v.to_string();
    assert_eq!(text.matches("\"forcing_extrapolation_allowed\"").count(), 2, "one per row: {v:#}");
}

/// Two ODE fits, each made under the override and each holding the forcing
/// when scored in compare's process: each row reports its own hold.
#[test]
fn compare_reports_each_ode_fits_hold() {
    let tmp = tempfile::tempdir().unwrap();
    let (a_dir, c_dir) = (tmp.path().join("a"), tmp.path().join("c"));
    std::fs::create_dir_all(&a_dir).unwrap();
    std::fs::create_dir_all(&c_dir).unwrap();
    let a = held_out_fit(&a_dir, "allow_forcing_extrapolation = true\n", true);
    let c = held_out_fit(&c_dir, "allow_forcing_extrapolation = true\n", true);
    let o = camdl(tmp.path(), &["compare", &a, &c]);
    assert!(o.status.success(), "{}", stderr(&o));
    let lines = held_lines(&o);
    assert_eq!(lines.len(), 2, "{lines:?}\n{}\n{}", String::from_utf8_lossy(&o.stdout), stderr(&o));
    assert_ne!(lines[0].split(':').next(), lines[1].split(':').next(), "each names its own row: {lines:?}");
    // Reported on the rows only, as for a fit a `camdl pfilter` child
    // scores: not also as an anonymous warning when each fit met it.
    assert!(!stderr(&o).contains("the run evaluated it"), "{}", stderr(&o));
}

/// `simulate --draws posterior --fit` replays a fit made under `[config]
/// allow_forcing_extrapolation` under it: the replay holds the forcing rather
/// than refusing it, says so once, and is the same computation as the replay
/// given the flag, so the flagged run adds no run id (gh#1235).
#[test]
fn simulate_replays_a_fit_under_its_configs_override() {
    let tmp = tempfile::tempdir().unwrap();
    let m = model(tmp.path(), "linear", &ROWS_TO_50);
    let o = fit_run(tmp.path(), &m, "allow_forcing_extrapolation = true\n");
    assert!(o.status.success(), "{}", stderr(&o));
    let leaf = method_leaf(tmp.path(), "pgas");
    let replay = |extra: &[&str]| {
        let mut args = vec![
            "simulate", m.to_str().unwrap(), "--draws", "posterior", "--fit", &leaf, "--n-draws", "2",
            "--seed", "1",
        ];
        args.extend_from_slice(extra);
        camdl(tmp.path(), &args)
    };
    let o = replay(&[]);
    let err = stderr(&o);
    assert!(o.status.success(), "the fit's override lets the replay run:\n{err}");
    assert_eq!(warnings(&err), 1, "one warning for the one forcing:\n{err}");
    let after = run_ids(&tmp.path().join("cas"));
    let o = replay(&["--allow-forcing-extrapolation"]);
    assert!(o.status.success(), "under the flag too:\n{}", stderr(&o));
    assert_eq!(run_ids(&tmp.path().join("cas")), after, "the fit's override and the flag are one computation");
}

/// A refused fit does not leave its claim open: the method's `run.json` is
/// marked `failed`, with the refusal as its failure, and no lock is left
/// behind, for each method that holds a claim while it runs.
#[test]
fn a_refused_fit_marks_its_claim_failed() {
    refused_fit_marks_its_claim_failed(1);
}

/// With two chains, the chain that meets the refusal ends the run and the
/// other chain stops at its next poll: it does not go on sweeping, so the
/// failed leaf holds no chain's resume state.
#[test]
fn a_refused_two_chain_fit_stops_every_chain() {
    let leaves = refused_fit_marks_its_claim_failed(2);
    for (algorithm, leaf, _tmp) in leaves {
        assert!(leaf.join("run.json").is_file(), "{algorithm}: the leaf {}", leaf.display());
        let mut stack = vec![leaf.clone()];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).unwrap().flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else {
                    assert!(p.file_name().is_none_or(|n| n != "resume_state.bin"),
                        "{algorithm}: a chain went on after the refusal and wrote {}", p.display());
                }
            }
        }
    }
}

/// The checks of [`a_refused_fit_marks_its_claim_failed`] for a fit of
/// `chains` chains, under each method that holds a claim while it runs;
/// returns each method's failed leaf.
fn refused_fit_marks_its_claim_failed(chains: usize) -> Vec<(&'static str, PathBuf, tempfile::TempDir)> {
    let mut leaves = Vec::new();
    for algorithm in ["pgas", "pmmh"] {
        let tmp = tempfile::tempdir().unwrap();
        let m = model(tmp.path(), "linear", &ROWS_TO_50);
        let toml = fit_toml(tmp.path(), &m, "");
        let src = std::fs::read_to_string(&toml).unwrap().replace("algorithm = \"pgas\"", &format!("algorithm = \"{algorithm}\""));
        let src = if algorithm == "pmmh" { src.replace("sweeps = 4", "iterations = 4") } else { src };
        let src = src.replace("chains = 1", &format!("chains = {chains}"));
        std::fs::write(&toml, src).unwrap();
        let o = camdl(tmp.path(), &["fit", "run", toml.to_str().unwrap()]);
        let err = stderr(&o);
        assert!(!o.status.success(), "{algorithm}: refused:\n{err}");
        assert!(err.contains("--allow-forcing-extrapolation"), "{algorithm}: the refusal:\n{err}");
        let records = statuses(&tmp.path().join("results"));
        let method: Vec<_> = records.iter().filter(|(_, v)| v["kind"] != "fit").collect();
        assert!(!method.is_empty(), "{algorithm}: the method leaf was claimed: {records:?}");
        for (path, v) in &method {
            assert_eq!(v["status"], "failed", "{algorithm}: {} must be failed, not left open:\n{v:#}", path.display());
            let failure = v.pointer("/failure/message").and_then(|m| m.as_str()).unwrap_or_default();
            assert!(failure.contains("forcing `mult`"), "{algorithm}: the failure records the refusal:\n{v:#}");
            assert!(!path.with_file_name(".lock").exists(), "{algorithm}: no lock left at {}", path.display());
        }
        leaves.push((algorithm, method[0].0.parent().unwrap().to_path_buf(), tmp));
    }
    leaves
}

/// Every `run.json` under `root`, with its path.
/// Every record under `root`: an old-store leaf's `run.json`, and for a
/// filter run in the run store its `run.json` with the `run_id` and `kind` of
/// its `identity.json` and, as `inputs`, its `provenance.json`.
fn statuses(root: &Path) -> Vec<(PathBuf, serde_json::Value)> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == "run.json") {
                let mut v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
                if let Some(id) = run_store_identity(&p) {
                    // The run store's content artifacts are inputs of a run,
                    // not runs.
                    if id["kind"] == "compiled_model" || id["kind"] == "data_table" {
                        continue;
                    }
                    v["run_id"] = id["run_id"].clone();
                    v["kind"] = id["kind"].clone();
                    let prov = p.with_file_name("provenance.json");
                    if let Ok(b) = std::fs::read_to_string(&prov) {
                        v["inputs"] = serde_json::from_str(&b).unwrap();
                    }
                }
                out.push((p, v));
            }
        }
    }
    out
}

/// Under `--allow-forcing-extrapolation` each kind runs, says so once, and
/// records the override in the stored cell's identity record.
#[test]
fn the_override_holds_with_one_warning_per_function() {
    for kind in KINDS {
        let tmp = tempfile::tempdir().unwrap();
        let m = model(tmp.path(), kind, &ROWS_TO_50);
        let o = simulate(tmp.path(), &m, &["--allow-forcing-extrapolation"]);
        let err = stderr(&o);
        assert!(o.status.success(), "{kind}: the override lets the run proceed:\n{err}");
        assert_eq!(warnings(&err), 1, "{kind}: exactly one warning:\n{err}");
        assert!(err.contains(WARNING) || err.contains("warning"), "{kind}: it is a warning:\n{err}");
        assert!(err.contains("held from t = 50 to the end of the run"), "{kind}: names the span held:\n{err}");
        assert!(!err.contains("t = 51"), "{kind}: names no time outside the range:\n{err}");
        let recorded = recorded_overrides(&tmp.path().join("cas"));
        assert_eq!(recorded, vec![true], "{kind}: the override is recorded in the cell's identity");
    }
}

/// The held values are the values the forcing had before this check existed:
/// for the linear and piecewise-constant reads, a hold at the last data point
/// is the same function as data extended by one more row carrying the last
/// value, so the two trajectories are byte-identical. (A cubic spline or a
/// B-spline changes shape inside its range when a point is added, so they
/// have no such oracle; the hold itself is untouched code.)
#[test]
fn the_held_trajectory_is_the_extended_data_trajectory() {
    for kind in ["linear", "constant"] {
        let held = tempfile::tempdir().unwrap();
        let m = model(held.path(), kind, &ROWS_TO_50);
        let o = simulate(held.path(), &m, &["--allow-forcing-extrapolation"]);
        assert!(o.status.success(), "{kind}:\n{}", stderr(&o));

        let extended = tempfile::tempdir().unwrap();
        let mut rows = ROWS_TO_50.to_vec();
        rows.push((100, 0.5));
        let m = model(extended.path(), kind, &rows);
        let o = simulate(extended.path(), &m, &[]);
        let err = stderr(&o);
        assert!(o.status.success(), "{kind}: data covering the run need no override:\n{err}");
        assert_eq!(warnings(&err), 0, "{kind}: no warning in range:\n{err}");

        let a = std::fs::read(held.path().join("traj.tsv")).unwrap();
        let b = std::fs::read(extended.path().join("traj.tsv")).unwrap();
        assert!(a == b, "{kind}: the held trajectory must equal the extended-data one byte for byte");
    }
}

/// The override is identity: a run under it and the same run without it are
/// two runs, so a stored run is never served to the other, whether or not it
/// held a forcing. Here the forcing covers the run, so both runs succeed and
/// the second must be a cache miss, not a hit on the first one's leaf; a run
/// past the range without the override is refused before the store is read,
/// so it could not test identity (`simulate_refuses_each_kind_past_its_range`
/// covers that refusal). Profile's identity is tested by
/// `a_stored_held_profile_point_is_not_served_without_the_override`, whose
/// refusal comes from its cache scan.
#[test]
fn the_override_is_in_the_run_identity() {
    let tmp = tempfile::tempdir().unwrap();
    let m = model(tmp.path(), "bspline", &[(0, 1.0), (100, 1.0)]);
    let o = simulate(tmp.path(), &m, &["--allow-forcing-extrapolation"]);
    assert!(o.status.success(), "{}", stderr(&o));
    let o = simulate(tmp.path(), &m, &[]);
    assert!(o.status.success(), "{}", stderr(&o));
    assert_eq!(run_ids(&tmp.path().join("cas")).len(), 2,
        "simulate: the run without the override is a cache miss on the run with it");

    let data = tmp.path().join("cases.tsv");
    let mut body = String::from("time\tcases\n");
    for w in 1..=10 {
        body.push_str(&format!("{}\t{}\n", 7 * w, 10 + 3 * w));
    }
    std::fs::write(&data, body).unwrap();
    let params = tmp.path().join("params.toml");
    let pfilter = |extra: &[&str]| {
        let mut args = vec![
            "pfilter", m.to_str().unwrap(), "--data", data.to_str().unwrap(),
            "--params", params.to_str().unwrap(), "--particles", "500", "--seed", "1", "--dt", "1",
        ];
        args.extend_from_slice(extra);
        camdl(tmp.path(), &args)
    };
    let before = run_ids(&tmp.path().join("cas")).len();
    let o = pfilter(&["--allow-forcing-extrapolation"]);
    assert!(o.status.success(), "{}", stderr(&o));
    let o = pfilter(&[]);
    assert!(o.status.success(), "{}", stderr(&o));
    assert_eq!(run_ids(&tmp.path().join("cas")).len(), before + 2,
        "pfilter: the run without the override is a cache miss on the run with it");

}

/// The distinct `run_id`s of the committed `run.json`s under `cas`.
/// The `identity.json` beside a run store's `run.json`, if it is one.
fn run_store_identity(run_json: &Path) -> Option<serde_json::Value> {
    let b = std::fs::read_to_string(run_json.with_file_name("identity.json")).ok()?;
    serde_json::from_str(&b).ok()
}

fn run_ids(cas: &Path) -> std::collections::BTreeSet<String> {
    let mut out = std::collections::BTreeSet::new();
    let mut stack = vec![cas.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == "run.json") {
                let v: serde_json::Value = match run_store_identity(&p) {
                    // The run store's content artifacts are inputs of a run,
                    // not runs.
                    Some(id) if id["kind"] == "compiled_model" || id["kind"] == "data_table" => continue,
                    Some(id) => id,
                    None => serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap(),
                };
                if let Some(id) = v.get("run_id").and_then(|i| i.as_str()) {
                    out.insert(id.to_string());
                }
            }
        }
    }
    out
}

/// The same under `profile`: a profile point stored under the override is not
/// served, as a cached start, to the same profile without it.
#[test]
fn a_stored_held_profile_point_is_not_served_without_the_override() {
    let tmp = tempfile::tempdir().unwrap();
    let m = model(tmp.path(), "linear", &ROWS_TO_50);
    let toml = fit_toml(tmp.path(), &m, "");
    let prof = tmp.path().join("prof");
    let args = |extra: &'static [&'static str]| {
        let mut v: Vec<String> = [
            "profile", m.to_str().unwrap(), "--fit", toml.to_str().unwrap(), "--sweep", "gamma=0.1,0.2",
            "--particles", "8", "--iterations", "2", "--n-starts", "1", "--seed", "1", "-o",
            prof.to_str().unwrap(),
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        v.extend(extra.iter().map(|s| s.to_string()));
        v
    };
    let with: Vec<String> = args(&["--allow-forcing-extrapolation"]);
    let o = camdl(tmp.path(), &with.iter().map(String::as_str).collect::<Vec<_>>());
    assert!(o.status.success(), "the override lets the profile run:\n{}", stderr(&o));
    let without: Vec<String> = args(&[]);
    let o = camdl(tmp.path(), &without.iter().map(String::as_str).collect::<Vec<_>>());
    let err = stderr(&o);
    assert!(!o.status.success(), "the stored held profile points must not be served:\n{err}");
    assert!(err.contains("--allow-forcing-extrapolation"), "{err}");
}

/// A forcing whose data start after the run does is refused too, naming the
/// side it was evaluated on.
#[test]
fn a_run_before_the_first_data_point_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let m = model(tmp.path(), "linear", &[(10, 1.0), (50, 0.8), (100, 0.6)]);
    let o = simulate(tmp.path(), &m, &[]);
    let err = stderr(&o);
    assert!(!o.status.success(), "{err}");
    assert!(err.contains("before its first data point") && err.contains("t = 10 to t = 100"), "{err}");
}

/// A lag is applied before the check: a forcing with `lag = 10 'days` whose
/// data start where the run does is read at t − 10 from the run's start, before
/// its first data point, and is refused there although its data cover the
/// run's span. Extending the data back by the lag covers it.
#[test]
fn a_lagged_forcing_whose_data_start_at_the_run_start_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let m = model(tmp.path(), "linear", &[(0, 1.0), (50, 0.8), (100, 0.6)]);
    let src = std::fs::read_to_string(&m).unwrap().replace("method = linear", "method = linear\n    lag = 10 'days");
    assert!(src.contains("lag = 10 'days"), "the lag was added");
    std::fs::write(&m, src).unwrap();
    let o = simulate(tmp.path(), &m, &[]);
    let err = stderr(&o);
    assert!(!o.status.success(), "the lag reads before the data at the run's start:\n{err}");
    assert!(err.contains("before its first data point") && err.contains("t = 0 to t = 100"), "{err}");
    assert!(err.contains("the forcing has a lag"), "names the lag:\n{err}");

    std::fs::write(tmp.path().join("mult.tsv"), "time\tmult\n-10\t1.0\n0\t1.0\n50\t0.8\n100\t0.6\n").unwrap();
    let o = simulate(tmp.path(), &m, &[]);
    let err = stderr(&o);
    assert!(o.status.success(), "data extended back by the lag cover the run:\n{err}");
    assert_eq!(warnings(&err), 0, "{err}");
}

/// The check reads the times evaluated, not the run's span: a forcing read
/// only under a condition that is false past its range raises nothing.
#[test]
fn a_forcing_read_only_inside_its_range_is_not_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let m = model(tmp.path(), "linear", &ROWS_TO_50);
    let src = std::fs::read_to_string(&m).unwrap().replace(
        "@ mult * beta * S * (I / N)",
        "@ (if t < 40 'days then mult else 1.0) * beta * S * (I / N)",
    );
    assert!(src.contains("if t < 40 'days"), "the rate was rewritten");
    std::fs::write(&m, src).unwrap();
    let o = simulate(tmp.path(), &m, &[]);
    let err = stderr(&o);
    assert!(o.status.success(), "{err}");
    assert_eq!(warnings(&err), 0, "{err}");
}

/// A B-spline window that covers the run, and data that do, are not affected.
#[test]
fn a_range_covering_the_run_is_untouched() {
    let tmp = tempfile::tempdir().unwrap();
    let m = model(tmp.path(), "bspline", &[(0, 1.0), (100, 1.0)]);
    let o = simulate(tmp.path(), &m, &[]);
    let err = stderr(&o);
    assert!(o.status.success(), "{err}");
    assert_eq!(warnings(&err), 0, "{err}");
}

/// The `allow_forcing_extrapolation` value each stored simulation cell under
/// `cas` records in its identity record's `config`.
fn recorded_overrides(cas: &Path) -> Vec<bool> {
    let mut out = Vec::new();
    let mut stack = vec![cas.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == "identity.json") {
                let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
                if v["kind"] != "sim" {
                    continue;
                }
                if let Some(b) = v.pointer("/identity/config/allow_forcing_extrapolation").and_then(|b| b.as_bool()) {
                    out.push(b);
                }
            }
        }
    }
    out
}
