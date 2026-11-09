//! Every command that rebuilds "the fitted model" from a fit applies the fit
//! config's own `scenario` / `enable` / `disable`.
//!
//! `fit run` scores the model with the fit's selection applied — toggleable
//! interventions are off unless enabled (spec §14.4) — so a fit config with
//! `enable = ["vax"]` estimates under `vax`. The commands that replay that fit
//! must replay the same model: `fit predict`'s `fitted` arm and one-step
//! filter, the contrast arms, `simulate --draws posterior --fit`, `pfilter
//! --fit`, and the prequential `compare` derives. Each once rebuilt the model
//! from the archived IR with no selection, so `vax` was silently off (or, in
//! the one-step filter, every declared intervention silently on).
//!
//! The model's `vax` moves 90% of S to R at t = 2 and `vax2` moves half of S
//! at t = 3; both are toggleable. Every assertion compares two runs that must
//! be the same model (or must not), not a printed message — except the
//! refusal and the skipped-contrast note, whose wording is the contract.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set under cargo test");
    let p = Path::new(&manifest).join("../../target/release/camdl");
    assert!(
        p.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test` (gh#105)",
        p.display()
    );
    p
}

fn camdlc_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set under cargo test");
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(p.exists(), "camdlc.exe missing: {} - run `make build`", p.display());
    p
}

struct Tmp(PathBuf);
impl Tmp {
    fn path(&self) -> &Path { &self.0 }
}
impl Drop for Tmp {
    fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
}

fn tempdir(tag: &str) -> Tmp {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let base = std::env::temp_dir().join(format!(
        "camdl_fitted_arm_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

const HEAD: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.01, 1.0]
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 10000]
}
transitions {
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected  = I
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 990  I = 10 }
simulate { from = 0 'days  to = 6 'days }
"#;

/// Both interventions declared. `same` toggles nothing and sets `gamma` to
/// its `[fixed]` value, so on top of a fit it must be the fitted model.
const TAIL_BOTH: &str = r#"
interventions {
  vax  : transfer(fraction = 0.9, from = S, to = R) at [2]
  vax2 : transfer(fraction = 0.5, from = S, to = R) at [3]
}
scenarios {
  vax   { enable = [vax] }
  same  { set = { gamma = 0.1 } }
  novax { disable = [vax] }
}
"#;

/// Only `vax` declared: under a fit enabling `vax` this is the same fitted
/// model as [`TAIL_BOTH`] (its `vax2` is filtered out before anything runs).
const TAIL_VAX_ONLY: &str = r#"
interventions {
  vax  : transfer(fraction = 0.9, from = S, to = R) at [2]
}
scenarios {
  vax { enable = [vax] }
}
"#;

const CASES_TSV: &str = "time\tcases\n1\t10\n2\t11\n3\t10\n4\t12\n5\t11\n6\t10\n";

fn compile(dir: &Path, name: &str, src: &str) -> PathBuf {
    let model_src = dir.join(format!("{name}.camdl"));
    std::fs::write(&model_src, src).unwrap();
    let out = Command::new(camdlc_bin()).env_remove("CLICOLOR_FORCE").arg(&model_src).output().unwrap();
    assert!(out.status.success(), "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    let ir = dir.join(format!("{name}.ir.json"));
    std::fs::write(&ir, &out.stdout).unwrap();
    ir
}

/// A fit config `<name>.toml` over `ir`, with `top` spliced above `[model]`.
fn fit_config(dir: &Path, name: &str, ir: &Path, top: &str, method: &str) -> PathBuf {
    let data = dir.join("cases.tsv");
    std::fs::write(&data, CASES_TSV).unwrap();
    let fit = dir.join(format!("{name}.toml"));
    std::fs::write(&fit, format!(r#"output_dir = "{out}"
{top}
[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
[estimate]
beta = {{ bounds = [0.01, 1.0], start = 0.5, prior = {{ uniform = {{}} }} }}
[fixed]
gamma = 0.1
N0 = 1000
{method}
"#, out = dir.join(format!("res_{name}")).display(), ir = ir.display(),
        data = data.display())).unwrap();
    fit
}

const PMMH: &str = r#"[method]
algorithm = "pmmh"
backend = "chain_binomial"
chains = 1
particles = 50
iterations = 60
burn_in = 20
"#;

fn run(dir: &Path, args: &[&str]) -> Output {
    run_in(dir, &dir.join("store"), args)
}

/// [`run`] with the store root a handle (`@label`) resolves under.
fn run_in(dir: &Path, store: &Path, args: &[&str]) -> Output {
    Command::new(camdl_bin()).env_remove("CLICOLOR_FORCE")
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_OUTPUT_DIR", store)
        .args(args)
        .output()
        .expect("spawn camdl")
}

fn ok(out: Output, what: &str) -> Output {
    assert!(out.status.success(), "{what} failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    out
}

/// The one completed fit leaf (the method leaf holding `draws.tsv`) under `root`.
fn draws_leaf(root: &Path) -> PathBuf {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        if d.join("draws.tsv").is_file() && d.join("run.json").is_file() {
            found.push(d.clone());
        }
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            if e.path().is_dir() { stack.push(e.path()); }
        }
    }
    assert_eq!(found.len(), 1, "expected one fit leaf under {}: {found:?}", root.display());
    found.remove(0)
}

/// The rows of a `fit predict` stream file for one `scenario` and `horizon`,
/// scenario column dropped, so two arms' bands compare directly.
fn predict_rows(leaf: &Path, scenario: &str, horizon: &str) -> Vec<String> {
    let seg = leaf.parent().unwrap().parent().unwrap();
    let mut files: Vec<PathBuf> = std::fs::read_dir(seg.join("predict")).unwrap()
        .flatten().map(|e| e.path().join("predictive/cases.tsv"))
        .filter(|p| p.is_file()).collect();
    assert_eq!(files.len(), 1, "expected one predictive/cases.tsv: {files:?}");
    let text = std::fs::read_to_string(files.remove(0)).unwrap();
    let rows: Vec<String> = text.lines()
        .filter(|l| !l.starts_with('#'))
        .filter(|l| {
            let c: Vec<&str> = l.split('\t').collect();
            c.first() == Some(&scenario) && c.contains(&horizon)
        })
        .map(|l| l.split('\t').skip(1).collect::<Vec<_>>().join("\t"))
        .collect();
    assert!(!rows.is_empty(), "no `{scenario}` `{horizon}` rows in:\n{text}");
    rows
}

#[test]
fn predict_fitted_arm_and_scenario_arms_carry_the_fits_selection() {
    let tmp = tempdir("predict");
    let ir = compile(tmp.path(), "m", &format!("{HEAD}{TAIL_BOTH}"));
    let fit = fit_config(tmp.path(), "f", &ir, r#"enable = ["vax"]"#, PMMH);
    ok(run(tmp.path(), &["fit", "run", &fit.to_string_lossy()]), "fit run");
    let leaf = draws_leaf(&tmp.path().join("res_f"));
    ok(run(tmp.path(), &["fit", "predict", &leaf.to_string_lossy(),
        "--horizon", "free_forward", "--scenario", "vax,same,novax"]), "fit predict");

    let fitted = predict_rows(&leaf, "fitted", "free_forward");
    // `vax` restates the fit's own selection, and the paired seeds are shared
    // across arms, so the two must be the same bands.
    assert_eq!(fitted, predict_rows(&leaf, "vax", "free_forward"),
        "the `fitted` arm of a fit with enable = [\"vax\"] must replay vax: it \
         differs from the `vax` arm");
    // A named arm that toggles nothing keeps the fit's interventions.
    assert_eq!(fitted, predict_rows(&leaf, "same", "free_forward"),
        "a scenario that toggles nothing must run on the fit's own selection");
    // An arm that disables vax is a different model.
    assert_ne!(fitted, predict_rows(&leaf, "novax", "free_forward"),
        "`novax` disables the fitted vax and must differ from the fitted arm");
}

#[test]
fn predict_one_step_filters_through_the_fitted_model() {
    // Two fits that are one fitted model: `vax2` is declared in one and off
    // (not enabled) in its fit. Same data, seed and method, so the posterior
    // is the same, and so must the one-step band be — the band filters the
    // data through the fitted model, which has no `vax2`.
    let tmp = tempdir("onestep");
    let ir_both = compile(tmp.path(), "both", &format!("{HEAD}{TAIL_BOTH}"));
    let ir_one = compile(tmp.path(), "one", &format!("{HEAD}{TAIL_VAX_ONLY}"));
    let f_both = fit_config(tmp.path(), "fb", &ir_both, r#"enable = ["vax"]"#, PMMH);
    let f_one = fit_config(tmp.path(), "fo", &ir_one, r#"enable = ["vax"]"#, PMMH);
    ok(run(tmp.path(), &["fit", "run", &f_both.to_string_lossy()]), "fit run fb");
    ok(run(tmp.path(), &["fit", "run", &f_one.to_string_lossy()]), "fit run fo");
    let (l_both, l_one) = (draws_leaf(&tmp.path().join("res_fb")),
                           draws_leaf(&tmp.path().join("res_fo")));
    assert_eq!(std::fs::read_to_string(l_both.join("draws.tsv")).unwrap(),
               std::fs::read_to_string(l_one.join("draws.tsv")).unwrap(),
        "precondition: the two fits are one fitted model and must share a posterior");
    for l in [&l_both, &l_one] {
        ok(run(tmp.path(), &["fit", "predict", &l.to_string_lossy(),
            "--horizon", "one_step"]), "fit predict one_step");
    }
    assert_eq!(predict_rows(&l_both, "fitted", "one_step"),
               predict_rows(&l_one, "fitted", "one_step"),
        "the one-step band ran an intervention the fit did not (vax2 is declared \
         but not enabled)");
}

#[test]
fn simulate_posterior_replays_the_fits_selection() {
    let tmp = tempdir("simulate");
    let ir = compile(tmp.path(), "m", &format!("{HEAD}{TAIL_BOTH}"));
    let fit = fit_config(tmp.path(), "f", &ir, r#"enable = ["vax"]"#, PMMH);
    ok(run(tmp.path(), &["fit", "run", &fit.to_string_lossy()]), "fit run");
    let leaf = draws_leaf(&tmp.path().join("res_f"));
    // The same draws outside any fit: no fit selection to apply, so the model
    // runs as written. Run FIRST, into the same store: its cells have the
    // parameters and seeds of the fit's implicit arm, and differ only in the
    // fit's selection — which must therefore key the fit's cells apart, or
    // the store serves this run's vax-off trajectories for them.
    let bare = tmp.path().join("bare");
    std::fs::create_dir_all(&bare).unwrap();
    std::fs::copy(leaf.join("draws.tsv"), bare.join("draws.tsv")).unwrap();
    let sim_from = |from: &Path, out: &str, extra: &[&str]| {
        let mut args = vec!["simulate", ir.to_str().unwrap(), "--draws", "posterior",
            "--fit", from.to_str().unwrap(), "--n-draws", "5", "--seed", "3", "-o", out];
        args.extend_from_slice(extra);
        ok(run(tmp.path(), &args), "simulate");
        std::fs::read_to_string(tmp.path().join(out)).unwrap()
            .lines().filter(|l| !l.starts_with('#'))
            // the scenario column carries the arm's label; compare the numbers
            .map(|l| l.split('\t').filter(|c| !c.starts_with("(none)")).collect::<Vec<_>>().join("\t"))
            .collect::<Vec<_>>()
    };
    let sim = |out: &str, extra: &[&str]| sim_from(&leaf, out, extra);
    let as_written = sim_from(&bare, "bare.tsv", &[]);
    let implicit = sim("implicit.tsv", &[]);
    assert_ne!(implicit, as_written,
        "the fit's implicit arm (vax on) was served the stored vax-off cells of \
         the same draws: the fit's selection is not in the cell identity");
    assert_eq!(implicit, sim("vax.tsv", &["--enable", "vax"]),
        "the implicit arm of `--draws posterior --fit` must be the fitted model \
         (vax on), the same trajectories as `--enable vax`");
    assert_ne!(implicit, sim("novax.tsv", &["--disable", "vax"]),
        "`--disable vax` switches off the fitted vax and must differ");
}

#[test]
fn simulate_prior_and_file_draws_replay_the_fits_selection() {
    // `--draws prior --fit` and `--draws <file> --fit` read the fit config for
    // priors / `[fixed]`; they must also run its `enable`, as `--draws
    // posterior --fit` does. `f0` is the same fit with no selection: its
    // cells are the as-written model and run first into the same store, so a
    // selection missing from the cell identity would be served its cells.
    let tmp = tempdir("simulate_other_draws");
    let ir = compile(tmp.path(), "m", &format!("{HEAD}{TAIL_BOTH}"));
    let fit = fit_config(tmp.path(), "f", &ir, r#"enable = ["vax"]"#, PMMH);
    let f0 = fit_config(tmp.path(), "f0", &ir, "", PMMH);
    ok(run(tmp.path(), &["fit", "run", &fit.to_string_lossy()]), "fit run");
    let leaf = draws_leaf(&tmp.path().join("res_f"));
    std::fs::copy(leaf.join("draws.tsv"), tmp.path().join("d.tsv")).unwrap();
    let sim = |out: &str, mode: &[&str], fit: &Path, extra: &[&str]| {
        let mut args = vec!["simulate", ir.to_str().unwrap()];
        args.extend_from_slice(mode);
        args.extend_from_slice(&["--fit", fit.to_str().unwrap(), "--seed", "3", "-o", out]);
        args.extend_from_slice(extra);
        ok(run(tmp.path(), &args), "simulate");
        std::fs::read_to_string(tmp.path().join(out)).unwrap()
            .lines().filter(|l| !l.starts_with('#'))
            .map(|l| l.split('\t').filter(|c| !c.starts_with("(none)")).collect::<Vec<_>>().join("\t"))
            .collect::<Vec<_>>()
    };
    for (tag, mode) in [("prior", vec!["--draws", "prior", "-n", "4"]),
                        ("file", vec!["--draws", "d.tsv"])] {
        let o = |n: &str| format!("{tag}_{n}.tsv");
        let as_written = sim(&o("f0"), &mode, &f0, &[]);
        let implicit = sim(&o("implicit"), &mode, &fit, &[]);
        assert_ne!(implicit, as_written,
            "--draws {tag} --fit: the fit's selection (enable vax) is ignored — the \
             implicit arm is the as-written model");
        assert_eq!(implicit, sim(&o("vax"), &mode, &fit, &["--enable", "vax"]),
            "--draws {tag} --fit: the implicit arm must be the fitted model, the \
             same trajectories as `--enable vax`");
        assert_ne!(implicit, sim(&o("novax"), &mode, &fit, &["--disable", "vax"]),
            "--draws {tag} --fit: `--disable vax` switches off the fitted vax and \
             must differ");
    }
}

fn loglik(out: &Output) -> f64 {
    String::from_utf8_lossy(&out.stdout).lines()
        .find_map(|l| l.trim().parse::<f64>().ok())
        .unwrap_or_else(|| panic!("no loglik on stdout: {}", String::from_utf8_lossy(&out.stdout)))
}

#[test]
fn pfilter_fit_scores_under_the_fits_selection() {
    let tmp = tempdir("pfilter");
    let ir = compile(tmp.path(), "m", &format!("{HEAD}{TAIL_BOTH}"));
    let fit = fit_config(tmp.path(), "f", &ir, r#"enable = ["vax"]"#, PMMH);
    let data = format!("cases={}", tmp.path().join("cases.tsv").display());
    let theta = ["--param", "beta=0.5", "--param", "gamma=0.1", "--param", "N0=1000",
                 "--particles", "200", "--seed", "1"];
    let pf = |extra: &[&str]| {
        let mut args = vec!["pfilter", ir.to_str().unwrap()];
        args.extend_from_slice(&theta);
        args.extend_from_slice(extra);
        run(tmp.path(), &args)
    };
    let with_fit = loglik(&ok(pf(&["--fit", fit.to_str().unwrap()]), "pfilter --fit"));
    let with_vax = loglik(&ok(pf(&["--data", &data, "--enable", "vax"]), "pfilter --enable vax"));
    let bare = loglik(&ok(pf(&["--data", &data]), "pfilter"));
    assert_eq!(with_fit, with_vax,
        "pfilter --fit must score the fit's model (vax enabled): got {with_fit}, \
         --enable vax scores {with_vax}, no intervention scores {bare}");
    assert_ne!(with_fit, bare, "precondition: vax must move the likelihood");

    // The same selection restated on the command line is the same problem
    // (the fit config already has enable = [\"vax\"]), not a conflict.
    let restated = loglik(&ok(pf(&["--fit", fit.to_str().unwrap(), "--enable", "vax"]),
        "pfilter --fit with the fit's own --enable"));
    assert_eq!(restated, with_fit,
        "--enable vax beside a fit config that enables vax scores the fit's model");

    // A flag naming a different selection from the fit's is refused, as on
    // `survey --fit` / `profile --fit`: a run scores one problem.
    let out = pf(&["--fit", fit.to_str().unwrap(), "--disable", "vax"]);
    assert!(!out.status.success(), "pfilter --fit (enable vax) --disable vax must be refused");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("neither silently wins"), "unexpected refusal: {stderr}");
}

#[test]
fn compare_derives_the_prequential_under_the_fits_selection() {
    // The prequential `compare` derives from a fit is a `pfilter` at θ̂ on the
    // fit's archived model; it must run the fit's interventions.
    let tmp = tempdir("compare");
    let ir = compile(tmp.path(), "m", &format!("{HEAD}{TAIL_BOTH}"));
    let if2 = "[method]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 1\n\
               particles = 50\niterations = 3\ncooling = 0.7\n";
    let fa = fit_config(tmp.path(), "a", &ir, r#"enable = ["vax"]"#, if2);
    let fb = fit_config(tmp.path(), "b", &ir, "", if2);
    // Both fits in one store, so `@a` / `@b` resolve under it.
    let store = tmp.path().join("res_ab");
    for f in [&fa, &fb] {
        let text = std::fs::read_to_string(f).unwrap();
        let text = text.lines().map(|l| if l.starts_with("output_dir") {
            format!("output_dir = \"{}\"", store.display())
        } else { l.to_string() }).collect::<Vec<_>>().join("\n");
        std::fs::write(f, text).unwrap();
    }
    let run = |dir: &Path, args: &[&str]| run_in(dir, &store, args);
    ok(run(tmp.path(), &["fit", "run", &fa.to_string_lossy(), "--label", "a"]), "fit run a");
    ok(run(tmp.path(), &["fit", "run", &fb.to_string_lossy(), "--label", "b"]), "fit run b");
    let theta = ok(run(tmp.path(), &["fit", "summary", "@a", "--params-only"]),
        "fit summary --params-only");
    std::fs::write(tmp.path().join("theta.toml"), &theta.stdout).unwrap();
    let archived = std::fs::read_dir(store.join("fits")).unwrap().flatten()
        .map(|e| e.path().join("model.ir.json")).find(|p| p.is_file())
        .expect("fit a's archived model.ir.json");
    let data = format!("cases={}", tmp.path().join("cases.tsv").display());
    ok(run(tmp.path(), &["pfilter", archived.to_str().unwrap(), "--data", &data,
        "--params", "theta.toml", "--enable", "vax", "--save-prequential", "manual",
        "--particles", "300", "--seed", "7"]), "manual pfilter");
    let manual: sim::inference::prequential::PrequentialTrace = serde_json::from_str(
        &std::fs::read_to_string(tmp.path().join("manual.json")).unwrap()).unwrap();

    let out = ok(run(tmp.path(), &["compare", "@a", "@b",
        "--particles", "300", "--seed", "7", "--replicates", "1", "--draws", "1",
        "--format", "json"]), "compare");
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let derived = v["rows"].as_array().unwrap().iter()
        .find(|r| r["path"] == "@a")
        .unwrap_or_else(|| panic!("@a's row in {v}"))["elpd"].as_f64().unwrap();
    assert!((derived - manual.elpd()).abs() < 1e-9,
        "compare derived elpd {derived} for a fit with enable = [\"vax\"] must equal \
         a pfilter at θ̂ with --enable vax ({})", manual.elpd());
}

/// A PGAS fit that saves its smoothed paths, so `fit predict` emits contrasts.
const PGAS: &str = r#"[method]
algorithm = "pgas"
backend = "chain_binomial"
chains = 1
particles = 60
sweeps = 12
burn_in = 4
thin = 1
"#;

#[test]
fn contrast_arms_carry_the_fits_selection() {
    // `fitted` and `vax` are one model on a fit with enable = ["vax"]: no
    // intervention is toggled between them, so the contrast has no fork and is
    // skipped with its note. Dropping the fit's selection from the `fitted` arm
    // would toggle vax and band a spurious difference.
    let tmp = tempdir("contrasts");
    let src = format!("{HEAD}{TAIL_BOTH}\nquantities {{\n  total = final(R)\n}}\n\
        contrasts {{\n  restated = fitted.quantities.total - vax.quantities.total\n  \
        averted = novax.quantities.total - fitted.quantities.total\n}}\n");
    let ir = compile(tmp.path(), "m", &src);
    let fit = fit_config(tmp.path(), "f", &ir, r#"enable = ["vax"]"#, PGAS);
    ok(run(tmp.path(), &["fit", "run", &fit.to_string_lossy()]), "fit run");
    let leaf = draws_leaf(&tmp.path().join("res_f"));
    let out = ok(run(tmp.path(), &["fit", "predict", &leaf.to_string_lossy(),
        "--horizon", "free_forward"]), "fit predict");
    let stderr = String::from_utf8_lossy(&out.stderr);
    let seg = leaf.parent().unwrap().parent().unwrap();
    let pdir = std::fs::read_dir(seg.join("predict")).unwrap().flatten().next().unwrap().path();
    assert!(!pdir.join("contrasts/restated.tsv").exists(),
        "`fitted` and `vax` are one model under the fit's selection, so `restated` \
         has no toggled intervention and must not be banded:\n{stderr}");
    assert!(pdir.join("contrasts/averted.tsv").is_file(),
        "`novax` toggles the fitted vax off, so `averted` must be banded:\n{stderr}");
}
