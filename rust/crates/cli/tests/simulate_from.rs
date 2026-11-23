//! A run's start chosen from the data and per run: `simulate { from }` as an
//! observation anchor, `camdl simulate --from`, and a fit config's
//! `[simulate] from`, end to end through the binary.
//!
//! The motivating case is two models identical except their start (one per
//! data window). With these, one model serves both: the window is a property
//! of the run. What is pinned here:
//!
//!  1. `from = first_obs` resolves against the bound data, to the start of the
//!     earliest observed period.
//!  2. Precedence: `--from` over `[simulate] from` over the model's `from`.
//!  3. Identity: one resolved start reached by different spellings is one run
//!     (and one fit, whether the start is written in the model or the config);
//!     different starts are different runs.
//!  4. An anchored start with no data is refused by name.
//!  5. `pfilter --fit` and `camdl predict` use the start the config gives.

use std::path::{Path, PathBuf};
mod fit_store_support;
mod sim_store_cells;

use std::process::{Command, Output};

fn bin() -> PathBuf {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/release/camdl");
    assert!(
        p.exists(),
        "release camdl binary missing: {} — run `make build-rust` or `make test` (gh#105)",
        p.display()
    );
    p
}

/// SIR observed weekly. The `init {}` reads `t`, so where the run starts is
/// visible in the initial state as well as in the first trajectory row.
fn model_body(from: &str) -> String {
    format!(
        r#"time_unit = 'days

compartments {{ S, I, R }}

parameters {{
  beta  : rate in [0.05, 2.0]
  gamma : rate in [0.01, 1.0]
}}

let N = S + I + R

transitions {{
  infection : S --> I @ beta * S * (I / N)
  recovery  : I --> R @ gamma * I
}}

init {{ S = 990 'count + t / (1 'days) * 1 'count  I = 10 }}

observations {{
  cases {{
    columns       {{ time : time, cases : count }}
    covers        = closing_at(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    cases         ~ poisson(rate = projected)
  }}
}}

output {{ trajectories {{ every = 7 'days }} }}

simulate {{ from = {from}  to = 100 'days  dt = 1 'days }}
"#
    )
}

/// Weekly counts whose first row is labelled day 12: under
/// `closing_at(time, 7 'days)` that row covers days 5 to 12, so
/// `first_obs` = 5.
fn write_data(path: &Path) {
    let mut body = String::from("time\tcases\n");
    for i in 0..10 {
        body.push_str(&format!("{}\t{}\n", 12 + 7 * i, 3 + i));
    }
    std::fs::write(path, body).unwrap();
}

struct Project {
    dir: PathBuf,
    model: PathBuf,
}

/// A directory holding `m.camdl` (with the given `from`), `cases.tsv` and
/// `params.toml`. Every project's model file has the same name, so two
/// projects' models differ only in what their text says.
fn project(root: &Path, name: &str, from: &str) -> Project {
    let dir = root.join(name);
    std::fs::create_dir_all(&dir).unwrap();
    let model = dir.join("m.camdl");
    std::fs::write(&model, model_body(from)).unwrap();
    write_data(&dir.join("cases.tsv"));
    std::fs::write(dir.join("params.toml"), "beta = 0.12\ngamma = 0.1\n").unwrap();
    Project { dir, model }
}

/// A fit config beside the model, with an optional `[simulate]` table.
fn fit_toml(p: &Project, file: &str, simulate_from: Option<&str>) -> PathBuf {
    let simulate = simulate_from
        .map(|f| format!("[simulate]\nfrom = {f}\n"))
        .unwrap_or_default();
    let path = p.dir.join(file);
    std::fs::write(
        &path,
        format!(
            r#"[model]
camdl = "m.camdl"
[data.observations]
cases = "cases.tsv"
{simulate}
[estimate]
beta = {{ bounds = [0.05, 2.0], start = 0.12, prior = {{ uniform = {{}} }} }}
[fixed]
gamma = 0.1
[method]
algorithm  = "pmmh"
backend    = "chain_binomial"
chains     = 1
particles  = 20
iterations = 20
burn_in    = 5
thin       = 1
"#
        ),
    )
    .unwrap();
    path
}

/// A config's file name, run from its own directory: the fit identity keeps
/// the model and data paths as the command line resolves them, so two
/// projects compare equal only when both are named relatively.
fn file_name(p: &Path) -> &str {
    p.file_name().unwrap().to_str().unwrap()
}

fn camdl(p: &Project, args: &[&str]) -> Output {
    camdl_in(p, args, Some(&p.dir.join("cas")))
}

/// As [`camdl`], with the run store under `store` given, or (`None`) the
/// project's own, `results/`.
fn camdl_in(p: &Project, args: &[&str], store: Option<&Path>) -> Output {
    let mut c = Command::new(bin());
    if let Some(s) = store {
        c.env("CAMDL_STORE", std::path::Path::new(&s).join("camdl-store"));
    } else {
        c.env("CAMDL_STORE", p.dir.join("results"));
    }
    c
        .env_remove("CLICOLOR_FORCE")
        .current_dir(&p.dir)
        .args(args)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_IR_CACHE_DIR", p.dir.join("irc"))
        .output()
        .expect("spawn camdl")
}

fn ok(o: &Output, what: &str) -> String {
    let stderr = String::from_utf8_lossy(&o.stderr).into_owned();
    assert!(o.status.success(), "{what} must succeed:\n{stderr}");
    stderr
}

/// `simulate`, returning (stderr, the first trajectory row's `t` and `S`).
fn simulate(p: &Project, extra: &[&str]) -> (String, f64, f64) {
    let out = p.dir.join("traj.tsv");
    let mut args = vec!["simulate", "m.camdl", "--params", "params.toml", "--seed", "1",
                        "-o", out.to_str().unwrap()];
    args.extend_from_slice(extra);
    let stderr = ok(&camdl(p, &args), &format!("simulate {extra:?}"));
    let text = std::fs::read_to_string(&out).unwrap();
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    let row: Vec<&str> = lines.next().unwrap().split('\t').collect();
    let col = |n: &str| row[header.iter().position(|h| *h == n).unwrap()].parse::<f64>().unwrap();
    (stderr, col("t"), col("S"))
}

/// Every simulation cell's run id stored under the project's store.
fn run_ids(p: &Project) -> Vec<String> {
    sim_store_cells::sim_run_ids(&p.dir.join("cas").join("camdl-store"))
}

#[test]
fn an_anchored_start_resolves_to_the_first_observed_period() {
    let tmp = tempfile::tempdir().unwrap();
    let p = project(tmp.path(), "a", "first_obs");
    let toml = fit_toml(&p, "fit.toml", None);
    let (stderr, t0, s0) = simulate(&p, &["--fit", toml.to_str().unwrap()]);
    assert!(stderr.contains("simulate { from }: first_obs → t = 5"), "{stderr}");
    assert_eq!(t0, 5.0, "the run starts where the first observed week opens");
    assert_eq!(s0, 995.0, "init {{}} is evaluated at the resolved start (990 + t)");

    // The offset form moves it, through the same resolution.
    std::fs::write(&p.model, model_body("first_obs - 7 'days")).unwrap();
    let (_, t0, s0) = simulate(&p, &["--fit", toml.to_str().unwrap()]);
    assert_eq!((t0, s0), (-2.0, 988.0));
}

#[test]
fn an_anchored_start_with_no_data_is_refused_by_name() {
    let tmp = tempfile::tempdir().unwrap();
    let p = project(tmp.path(), "a", "first_obs");
    let o = camdl(&p, &["simulate", "m.camdl", "--params", "params.toml"]);
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(!o.status.success(), "an unresolvable start must be refused:\n{stderr}");
    assert!(stderr.contains("simulate { from = first_obs }") && stderr.contains("--fit"),
        "the refusal names the anchored start and where data comes from:\n{stderr}");

    // An anchored --from on a literal model, likewise.
    let q = project(tmp.path(), "b", "0 'days");
    let o = camdl(&q, &["simulate", "m.camdl", "--params", "params.toml", "--from", "first_obs"]);
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(!o.status.success(), "{stderr}");
    assert!(stderr.contains("--from \"first_obs\" is anchored"), "{stderr}");

    // pfilter with no data names it too.
    let o = camdl(&p, &["pfilter", "m.camdl", "--params", "params.toml", "--particles", "10"]);
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(!o.status.success() && stderr.contains("from = first_obs"), "{stderr}");

    // A literal --from replaces the anchor, so no data is needed.
    let (_, t0, _) = simulate(&p, &["--from", "3"]);
    assert_eq!(t0, 3.0);
}

#[test]
fn from_flag_over_config_over_model() {
    let tmp = tempfile::tempdir().unwrap();
    let p = project(tmp.path(), "a", "0 'days");
    let with_cfg = fit_toml(&p, "fit.toml", Some("\"first_obs\""));
    let cfg = with_cfg.to_str().unwrap();

    let (_, t_model, _) = simulate(&p, &[]);
    assert_eq!(t_model, 0.0, "the model's own start");
    let (stderr, t_cfg, _) = simulate(&p, &["--fit", cfg]);
    assert!(stderr.contains("[simulate] from"), "{stderr}");
    assert_eq!(t_cfg, 5.0, "the config's [simulate] from over the model");
    let (stderr, t_cli, _) = simulate(&p, &["--fit", cfg, "--from", "first_obs - 1 week"]);
    assert!(stderr.contains("(--from)"), "{stderr}");
    assert_eq!(t_cli, -2.0, "--from over the config");
}

#[test]
fn one_start_by_two_spellings_is_one_run_and_two_starts_are_two() {
    let tmp = tempfile::tempdir().unwrap();
    let p = project(tmp.path(), "a", "first_obs");
    let toml = fit_toml(&p, "fit.toml", None);
    let fit = toml.to_str().unwrap();

    simulate(&p, &["--fit", fit]); // the model's anchor: t = 5
    simulate(&p, &["--from", "5"]); // a literal start needs no data
    simulate(&p, &["--fit", fit, "--from", "first_obs"]);
    simulate(&p, &["--fit", fit, "--from", "first_obs + 0 days"]);
    assert_eq!(run_ids(&p).len(), 1, "one resolved start is one run: {:?}", run_ids(&p));

    simulate(&p, &["--from", "-2"]);
    assert_eq!(run_ids(&p).len(), 2, "a different start is a different run: {:?}", run_ids(&p));

    // Across model files: a model written with `from = 5 'days` is the run the
    // anchored model resolves to.
    let q = project(tmp.path(), "b", "5 'days");
    let o = Command::new(bin()).env_remove("CLICOLOR_FORCE").current_dir(&q.dir)
        .args(["simulate", "m.camdl", "--params", "params.toml", "--seed", "1", "-o", "t.tsv"])
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_IR_CACHE_DIR", q.dir.join("irc"))
        .env("CAMDL_STORE", std::path::Path::new(&p.dir.join("cas")).join("camdl-store"))
        .output().unwrap();
    ok(&o, "the literal-start model");
    assert_eq!(run_ids(&p).len(), 2,
        "`from = 5 'days` written in the model keys the same run as the anchor resolved to 5: {:?}",
        run_ids(&p));
}

/// The start a config gives is the fit's, by value: `[simulate] from` over an
/// anchored model and a model written with that start as a literal are one
/// fit, and `camdl predict` replays them identically. `pfilter --fit` scores the
/// same model. A different start is a different fit.
#[test]
fn a_config_start_is_one_fit_with_the_literal_and_replays() {
    let tmp = tempfile::tempdir().unwrap();
    let a = project(tmp.path(), "a", "first_obs");
    let b = project(tmp.path(), "b", "-2 'days");
    let fit_a = fit_toml(&a, "fit.toml", Some("\"first_obs - 7 days\""));
    let fit_b = fit_toml(&b, "fit.toml", None);
    let fit_a5 = fit_toml(&a, "fit5.toml", None);

    // The fit run of `<stem>.toml` in the project's store: the run whose
    // `fit.toml`, the file as given, is that file's bytes.
    let segment = |p: &Project, stem: &str| -> PathBuf {
        let text = std::fs::read(p.dir.join(format!("{stem}.toml"))).unwrap();
        fit_store_support::runs(&p.dir.join("results"))
            .into_iter()
            .find(|r| std::fs::read(r.join("fit.toml")).ok().as_deref() == Some(&text[..]))
            .unwrap_or_else(|| panic!("no fit run of {stem}.toml"))
    };
    let hash = |run: &Path| fit_store_support::run_inputs(run)["problem_id"].as_str().unwrap().to_string();

    ok(&camdl_in(&a, &["fit", "run", file_name(&fit_a), "--seed", "1"], None), "fit a");
    ok(&camdl_in(&b, &["fit", "run", file_name(&fit_b), "--seed", "1"], None), "fit b");
    ok(&camdl_in(&a, &["fit", "run", file_name(&fit_a5), "--seed", "1"], None), "fit a5");
    let (sa, sb, sa5) = (segment(&a, "fit"), segment(&b, "fit"), segment(&a, "fit5"));
    assert_eq!(hash(&sa), hash(&sb),
        "[simulate] from resolving to -2 and `from = -2 'days` in the model are one fit");
    assert_ne!(hash(&sa), hash(&sa5), "start -2 and start 5 are two fits");

    let problem: toml::Table = std::fs::read_to_string(fit_store_support::problem_of(&sa).join("problem.toml"))
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(problem["simulate"]["from"].as_float(), Some(-2.0), "the fit stores the start it ran from");

    let predict = |p: &Project, seg: &Path| -> String {
        ok(&camdl_in(p, &["predict", seg.to_str().unwrap(), "--n-draws", "3", "--seed", "1",
                                   "--mode", "free_forward"], None),
           "predict");
        let dir = fit_store_support::predictions(seg).pop().expect("the prediction");
        std::fs::read_to_string(dir.join("predictive/cases.tsv")).unwrap()
    };
    assert_eq!(predict(&a, &sa), predict(&b, &sb),
        "the stored fit replays its start: a's prediction equals the literal model's");

    let loglik = |p: &Project, fit: &Path| -> String {
        let o = camdl_in(p, &["pfilter", "m.camdl", "--fit", file_name(fit),
                           "--params", "params.toml", "--particles", "50", "--seed", "3"], None);
        let stdout = String::from_utf8_lossy(&o.stdout).into_owned();
        ok(&o, "pfilter --fit");
        stdout
    };
    let (la, lb, la5) = (loglik(&a, &fit_a), loglik(&b, &fit_b), loglik(&a, &fit_a5));
    assert_eq!(la, lb, "pfilter --fit scores the start the config gives");
    assert_ne!(la, la5, "and a different start scores differently");
}

#[test]
fn moving_a_literal_start_past_a_defaulted_recurring_schedule_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let p = project(tmp.path(), "a", "0 'days");
    let body = model_body("0 'days").replace(
        "output {",
        "interventions {\n  campaign : transfer(fraction = 0.1, from = S, to = R) {\n    \
         every = 30 'days\n    to = 90 'days\n  }\n}\n\noutput {",
    );
    std::fs::write(&p.model, body).unwrap();
    let o = camdl(&p, &["simulate", "m.camdl", "--params", "params.toml", "--from", "7",
                        "--enable", "campaign"]);
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(!o.status.success(), "an ambiguous schedule start must be refused:\n{stderr}");
    assert!(stderr.contains("campaign") && stderr.contains("first_obs"), "{stderr}");
}
