//! End-to-end tests that a fit keeps an estimated parameter inside its search
//! box, and refuses the box configurations it cannot honour.
//!
//! Every fixture is a closed SIR whose estimated parameter `c` appears nowhere
//! in the dynamics or the observation model, so the likelihood is flat in `c`
//! and its posterior is its prior `normal(0, 1)` restricted to the box. A
//! sampler that does not enforce the box puts the prior's mass outside the box
//! outside it (32% for `[-1, 1]`); one that enforces it puts none there.
//!
//!   1. `a_transform_key_in_estimate_is_refused_at_load` — `[estimate]` has
//!      no `transform` key; a config that carries one, even the map the
//!      parameter would get anyway, is refused under PMMH, MH and IF2 naming
//!      the key, and no draw is written.
//!   2. `a_kindless_estimated_parameter_is_refused` — a parameter with no
//!      `param_kind` (IR predating the field) has no known support, so
//!      estimating it is refused under PMMH, MH and IF2, naming it.
//!   3. `duration_box_straddling_zero_reaches_its_negative_half` (gh#967) — a
//!      `duration` declared `in [-1, 1]` got the log transform, which cannot
//!      represent a negative value; MH must now put about half its draws
//!      below zero.
//!   4. `prior_drawn_starts_lie_inside_the_box` — `starts = "from_prior"`
//!      drew `normal(0, 1)` starts with no regard to the box `[0, 1]`.
//!   6. `log_normal_on_a_positive_sub_box_is_the_truncated_log_normal`
//!      (support-keyed transforms §12.2) — a `rate` with no declared range,
//!      `~ log_normal(-1, 0.5)`, fit box [0.05, 1], under MH: the parameter
//!      takes `LogInterval`, and the draws must be the log-normal truncated
//!      to the box.
//!   5. `hierarchical_leaf_box_that_cuts_the_prior_is_refused` — a
//!      `normal(mu, 1)` leaf boxed to `[1, 2]` biases `mu` under PMMH and MH,
//!      and is refused before sampling; IF2 reads no prior and runs; a
//!      `log_normal` leaf on a `rate` with the half-infinite box `[0, ∞)` runs.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set under cargo test");
    let p = Path::new(&manifest).join("../../target/release/camdl");
    assert!(p.exists(), "release camdl binary missing: {} - run `make build-rust` or `make test`", p.display());
    p
}

fn camdlc_bin() -> Option<PathBuf> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    if p.exists() { Some(p) } else { None }
}

struct Tmp(PathBuf);
impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn tempdir(tag: &str) -> Tmp {
    let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let base = std::env::temp_dir().join(format!("camdl_bounds_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

/// A closed SIR with the extra parameter lines `params` spliced into the
/// `parameters` block. `beta` and `gamma` are held by every fit.
fn model_src(params: &str) -> String {
    format!(
        r#"time_unit = 'days
compartments {{ S, I, R }}
parameters {{
{params}
  beta  : rate in [0.05, 1.0]
  gamma : rate in [0.01, 1.0]
}}
transitions {{
  infection : S --> I @ beta * S * I / (S + I + R)
  recovery  : I --> R @ gamma * I
}}
observations {{
  cases {{
    columns       {{ time : time, cases : count }}
    projected  = I
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }}
}}
init {{ S = 990  I = 10 }}
simulate {{ from = 0 'days  to = 6 'days }}
"#
    )
}

/// Compile `src` to `<dir>/<stem>.ir.json` and write the shared data file.
fn compile(dir: &Path, stem: &str, src: &str) -> PathBuf {
    let camdlc = camdlc_bin().expect("camdlc.exe present");
    let model = dir.join(format!("{stem}.camdl"));
    std::fs::write(&model, src).unwrap();
    let out = Command::new(&camdlc).arg(&model).output().unwrap();
    assert!(out.status.success(), "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    let ir = dir.join(format!("{stem}.ir.json"));
    std::fs::write(&ir, &out.stdout).unwrap();
    std::fs::write(dir.join("cases.tsv"), "time\tcases\n1\t10\n2\t11\n3\t12\n4\t12\n5\t12\n6\t11\n").unwrap();
    ir
}

const PMMH: &str = "algorithm = \"pmmh\"\nbackend = \"chain_binomial\"\nchains = 2\nparticles = 10\niterations = 2000\nburn_in = 200";
const MH: &str = "algorithm = \"mh\"\nbackend = \"ode\"\nchains = 2\niterations = 4000\nburn_in = 400";
const IF2: &str = "algorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 2\nparticles = 50\niterations = 20\ncooling = 0.95";

/// Write a fit toml estimating `estimate` (one or more `name = {...}` lines)
/// under `method`, run it, and return the output and the results directory.
fn run_fit(dir: &Path, ir: &Path, estimate: &str, method: &str, tag: &str) -> (Output, PathBuf) {
    let results = dir.join(format!("results_{tag}"));
    let toml = format!(
        "output_dir = \"{out}\"\n[model]\ncamdl = \"{ir}\"\n[data.observations]\ncases = \"{data}\"\n\
         [estimate]\n{estimate}\n[fixed]\nbeta = 0.3\ngamma = 0.25\n[method]\n{method}\n",
        out = results.file_name().unwrap().to_string_lossy(),
        ir = ir.file_name().unwrap().to_string_lossy(),
        data = "cases.tsv",
    );
    let path = dir.join(format!("fit_{tag}.toml"));
    std::fs::write(&path, toml).unwrap();
    let out = Command::new(camdl_bin())
        .args(["fit", "run"])
        .arg(&path)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .unwrap();
    (out, results)
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn find_all(root: &Path, file: &str) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == file) {
                found.push(p);
            }
        }
    }
    found.sort();
    found
}

/// Every value of column `col` in the TSV files named `file` under `root`.
fn column_values(root: &Path, file: &str, col: &str) -> Vec<f64> {
    let files = find_all(root, file);
    assert!(!files.is_empty(), "no {file} under {}", root.display());
    let mut vals = Vec::new();
    for f in files {
        let text = std::fs::read_to_string(&f).unwrap();
        let mut lines = text.lines().filter(|l| !l.starts_with('#'));
        let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
        let i = header.iter().position(|h| *h == col).unwrap_or_else(|| panic!("no column {col} in {}", f.display()));
        for l in lines {
            let cells: Vec<&str> = l.split('\t').collect();
            if let Some(v) = cells.get(i).and_then(|c| c.parse::<f64>().ok()) {
                vals.push(v);
            }
        }
    }
    vals
}

fn assert_inside(vals: &[f64], lo: f64, hi: f64, what: &str) {
    assert!(!vals.is_empty(), "{what}: no values");
    let outside: Vec<f64> = vals.iter().copied().filter(|v| *v < lo || *v > hi).collect();
    assert!(
        outside.is_empty(),
        "{what}: {} of {} values outside [{lo}, {hi}], e.g. {:?}",
        outside.len(),
        vals.len(),
        &outside[..outside.len().min(5)]
    );
}

const C_REAL: &str = "  c : real ~ normal(mu = 0.0, sigma = 1.0)";

#[test]
fn a_transform_key_in_estimate_is_refused_at_load() {
    if camdlc_bin().is_none() {
        return;
    }
    let tmp = tempdir("transform_key");
    let ir = compile(&tmp.0, "ident", &model_src("  c : rate ~ normal(mu = 0.0, sigma = 1.0)"));
    for (method, tag) in [(PMMH, "pmmh"), (MH, "mh"), (IF2, "if2")] {
        // `log` on a rate's [0, 1] box is the map the parameter gets without
        // the key; it is refused all the same, because the key is gone.
        let (out, results) = run_fit(
            &tmp.0, &ir, "c = { bounds = [0, 1], start = 0.5, rw_sd = 0.1, transform = \"log\" }",
            method, tag);
        let err = stderr(&out);
        assert!(!out.status.success(), "{tag}: a `transform` key must be refused; stderr=\n{err}");
        assert!(err.contains("`transform` was removed") && err.contains("Delete the line"), "{tag}: {err}");
        assert!(find_all(&results, "draws.tsv").is_empty(), "{tag}: refused before sampling");
        assert!(find_all(&results, "parameter_traces.tsv").is_empty(), "{tag}: refused before sampling");
    }
}

/// Strip `param_kind` from `name` in a compiled IR, as in IR written before
/// the field existed.
fn strip_kind(ir: &Path, name: &str) -> PathBuf {
    let mut v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(ir).unwrap()).unwrap();
    for p in v["model"]["parameters"].as_array_mut().unwrap() {
        if p["name"] == name {
            p.as_object_mut().unwrap().remove("param_kind");
        }
    }
    let out = ir.with_file_name("nokind.ir.json");
    std::fs::write(&out, serde_json::to_string(&v).unwrap()).unwrap();
    out
}

#[test]
fn a_kindless_estimated_parameter_is_refused() {
    if camdlc_bin().is_none() {
        return;
    }
    let tmp = tempdir("kindless");
    let ir = strip_kind(&compile(&tmp.0, "ident", &model_src(C_REAL)), "c");
    let est = "c = { bounds = [-1, 1], start = 0.5, rw_sd = 0.1 }";
    for (method, tag) in [(PMMH, "pmmh"), (MH, "mh"), (IF2, "if2")] {
        let (out, results) = run_fit(&tmp.0, &ir, est, method, tag);
        let err = stderr(&out);
        assert!(!out.status.success(), "{tag}: a kindless parameter must be refused; stderr=\n{err}");
        assert!(err.contains("records no kind for `c`"), "{tag}: {err}");
        assert!(find_all(&results, "draws.tsv").is_empty(), "{tag}: refused before sampling");
        assert!(find_all(&results, "parameter_traces.tsv").is_empty(), "{tag}: refused before sampling");
    }
}

/// gh#967: a `duration` whose box straddles zero got the log transform,
/// which maps every value to a positive one, so the negative half of the box
/// was unreachable. Under a flat likelihood and a `normal(0, 1)` prior the
/// posterior puts half its mass below zero.
#[test]
fn duration_box_straddling_zero_reaches_its_negative_half() {
    if camdlc_bin().is_none() {
        return;
    }
    let tmp = tempdir("gh967");
    // The signed range is declared in the model: a fit-config box may only
    // narrow a `duration`'s `[0, ∞)` support where nothing is declared.
    let ir = compile(&tmp.0, "phase",
        &model_src("  c : duration in [-1.0, 1.0] ~ normal(mu = 0.0, sigma = 1.0)"));
    let (out, results) = run_fit(&tmp.0, &ir, "c = { bounds = [-1, 1], start = -0.5 }", MH, "mh");
    assert!(out.status.success(), "stderr=\n{}", stderr(&out));
    let vals = column_values(&results, "draws.tsv", "c");
    assert_inside(&vals, -1.0, 1.0, "mh");
    let below = vals.iter().filter(|v| **v < 0.0).count() as f64 / vals.len() as f64;
    assert!((0.3..=0.7).contains(&below), "fraction of draws below zero = {below}; expected about 0.5");
}

#[test]
fn prior_drawn_starts_lie_inside_the_box() {
    if camdlc_bin().is_none() {
        return;
    }
    let tmp = tempdir("starts");
    let ir = compile(&tmp.0, "ident", &model_src(C_REAL));
    let est = "c = { bounds = [0, 1] }";
    let pmmh = "algorithm = \"pmmh\"\nbackend = \"chain_binomial\"\nchains = 8\nparticles = 10\n\
                iterations = 20\nburn_in = 2\nstarts = \"from_prior\"";
    let if2 = "algorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 8\nparticles = 20\n\
               iterations = 2\ncooling = 0.95\nstarts = \"from_prior\"";
    for (method, tag) in [(pmmh, "pmmh"), (if2, "if2")] {
        let (out, results) = run_fit(&tmp.0, &ir, est, method, tag);
        assert!(out.status.success(), "{tag}: stderr=\n{}", stderr(&out));
        assert_inside(&column_values(&results, "chain_starts.tsv", "c"), 0.0, 1.0, tag);
    }
}

const HIER: &str = "  mu : real ~ normal(mu = 0.0, sigma = 1.0)
  a1 : real ~ normal(mu = mu, sigma = 1.0)
  a2 : real ~ normal(mu = mu, sigma = 1.0)";

#[test]
fn hierarchical_leaf_box_that_cuts_the_prior_is_refused() {
    if camdlc_bin().is_none() {
        return;
    }
    let tmp = tempdir("hier");
    let ir = compile(&tmp.0, "hier", &model_src(HIER));
    let est = "mu = { bounds = [-5, 5], start = 0.0 }\n\
               a1 = { bounds = [1, 2], start = 1.5 }\n\
               a2 = { bounds = [1, 2], start = 1.5 }";
    for (method, tag) in [(PMMH, "pmmh"), (MH, "mh")] {
        let (out, results) = run_fit(&tmp.0, &ir, est, method, tag);
        let err = stderr(&out);
        assert!(!out.status.success(), "{tag}: must be refused; stderr=\n{err}");
        assert!(err.contains("a1: bounds [1, 2], prior normal"), "{tag}: {err}");
        assert!(err.contains("a2: bounds [1, 2], prior normal"), "{tag}: {err}");
        assert!(err.contains("P(bounds | hyperparameter)"), "{tag}: {err}");
        assert!(err.contains("Widen the bounds to the prior's support, or remove the hierarchical prior"), "{tag}: {err}");
        assert!(find_all(&results, "draws.tsv").is_empty(), "{tag}: refused before sampling");
    }
    // IF2 reads no prior: the missing constant cannot bias it.
    let (out, _) = run_fit(&tmp.0, &ir, est, IF2, "if2");
    assert!(out.status.success(), "if2 must run; stderr=\n{}", stderr(&out));

    // Control: leaves with no box (an unconstrained `real` with no bounds has
    // none enforced) run under PMMH. A box with an infinite end gives no
    // auto `rw_sd`, so the step is explicit, and no box to spread starts in
    // (a hierarchical prior is not drawn from at init, so the default rule
    // would be `uniform_unconstrained`), so every chain starts at `start`.
    let open = "mu = { bounds = [-5, 5], start = 0.0 }\n\
                a1 = { start = 0.5, rw_sd = 0.1 }\na2 = { start = 0.5, rw_sd = 0.1 }";
    let pmmh_single = format!("{PMMH}\nstarts = \"single\"");
    let (out, _) = run_fit(&tmp.0, &ir, open, &pmmh_single, "pmmh_open");
    assert!(out.status.success(), "unboxed leaves must run; stderr=\n{}", stderr(&out));

    // Control: a `log_normal` leaf on a `rate` keeps the half-infinite box
    // [0, ∞), which is the family's whole support. That box has no region for
    // a box-drawing start rule either, so the chains start at `start`.
    let rate_src = "  m : real ~ normal(mu = 0.0, sigma = 1.0)
  r1 : rate ~ log_normal(mu = m, sigma = 1.0)
  r2 : rate ~ log_normal(mu = m, sigma = 1.0)";
    let rate_ir = compile(&tmp.0, "hier_rate", &model_src(rate_src));
    let rate_est = "m = { bounds = [-3, 3], start = 0.0 }\n\
                    r1 = { start = 1.0, rw_sd = 0.1 }\nr2 = { start = 1.0, rw_sd = 0.1 }";
    let (out, _) = run_fit(&tmp.0, &rate_ir, rate_est, &pmmh_single, "pmmh_rate");
    assert!(out.status.success(), "a rate leaf on [0, inf) must run; stderr=\n{}", stderr(&out));
    // The same leaf with a finite box cuts into (0, ∞) and is refused.
    let rate_boxed = "m = { bounds = [-3, 3], start = 0.0 }\nr1 = { bounds = [0.1, 5], start = 1.0 }\nr2 = { start = 1.0 }";
    let (out, _) = run_fit(&tmp.0, &rate_ir, rate_boxed, PMMH, "pmmh_rate_boxed");
    assert!(!out.status.success(), "a rate leaf on [0.1, 5] must be refused");
    assert!(stderr(&out).contains("r1: bounds [0.1, 5], prior log_normal"), "{}", stderr(&out));
}

/// Proposal §12.2, the companion of the interim fix's refusal of the box
/// [−1, 1]: the same prior, `c : rate ~ log_normal(-1, 0.5)` with no declared
/// range, on the fit box [0.05, 1], under MH. `c` takes `LogInterval` and
/// reads ln θ from its own coordinate, so the posterior (the likelihood is
/// flat in `c`) is the log-normal truncated to [0.05, 1]. Its mean and
/// standard deviation, by quadrature, must match the draws to 4 Monte Carlo
/// standard errors, each MCSE from batch means (20 batches per chain).
#[test]
fn log_normal_on_a_positive_sub_box_is_the_truncated_log_normal() {
    if camdlc_bin().is_none() {
        return;
    }
    let (mu, sigma, lo, hi) = (-1.0_f64, 0.5_f64, 0.05_f64, 1.0_f64);
    // The truncated law by quadrature on 200,000 midpoints.
    let n = 200_000;
    let h = (hi - lo) / n as f64;
    let (mut z, mut m1, mut m2) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let x = lo + (i as f64 + 0.5) * h;
        let u = (x.ln() - mu) / sigma;
        let d = (-0.5 * u * u).exp() / x;
        z += d;
        m1 += x * d;
        m2 += x * x * d;
    }
    let mean = m1 / z;
    let sd = (m2 / z - mean * mean).sqrt();

    let tmp = tempdir("lognormal_subbox");
    let ir = compile(&tmp.0, "ln", &model_src("  c : rate ~ log_normal(mu = -1.0, sigma = 0.5)"));
    let mh = "algorithm = \"mh\"\nbackend = \"ode\"\nchains = 4\niterations = 12000\nburn_in = 2000";
    let (out, results) = run_fit(&tmp.0, &ir, "c = { bounds = [0.05, 1], start = 0.3, rw_sd = 0.05 }", mh, "mh");
    assert!(out.status.success(), "stderr=\n{}", stderr(&out));
    let mut chains: Vec<Vec<f64>> = Vec::new();
    for f in find_all(&results, "draws.tsv") {
        let text = std::fs::read_to_string(&f).unwrap();
        let mut lines = text.lines().filter(|l| !l.starts_with('#'));
        let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
        let ic = header.iter().position(|h| *h == "c").unwrap();
        let ch = header.iter().position(|h| *h == "chain");
        for l in lines {
            let cells: Vec<&str> = l.split('\t').collect();
            let k: usize = ch.map_or(0, |i| cells[i].parse().unwrap());
            if chains.len() <= k { chains.resize(k + 1, Vec::new()); }
            chains[k].push(cells[ic].parse().unwrap());
        }
    }
    let all: Vec<f64> = chains.iter().flatten().copied().collect();
    assert_inside(&all, lo, hi, "mh");
    let mcse = |f: &dyn Fn(f64) -> f64| {
        let mut batch_means = Vec::new();
        for c in &chains {
            let b = c.len() / 20;
            for k in 0..20 {
                let s: f64 = c[k * b..(k + 1) * b].iter().map(|&x| f(x)).sum();
                batch_means.push(s / b as f64);
            }
        }
        let nb = batch_means.len() as f64;
        let m = batch_means.iter().sum::<f64>() / nb;
        (batch_means.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (nb - 1.0) / nb).sqrt()
    };
    let n_all = all.len() as f64;
    let got_mean = all.iter().sum::<f64>() / n_all;
    let got_m2 = all.iter().map(|x| x * x).sum::<f64>() / n_all;
    let got_sd = (got_m2 - got_mean * got_mean).sqrt();
    let se_mean = mcse(&|x| x);
    // sd = sqrt(E x² − (E x)²): delta-method MCSE through E x², with the
    // mean's error alongside it.
    let se_sd = (mcse(&|x| x * x) / (2.0 * got_sd)).hypot(se_mean * got_mean / got_sd);
    eprintln!("truncated log-normal [0.05, 1]: mean {got_mean:.4} (law {mean:.4}, MCSE {se_mean:.4}), \
               sd {got_sd:.4} (law {sd:.4}, MCSE {se_sd:.4})");
    assert!((got_mean - mean).abs() < 4.0 * se_mean, "mean {got_mean} vs {mean} (MCSE {se_mean})");
    assert!((got_sd - sd).abs() < 4.0 * se_sd, "sd {got_sd} vs {sd} (MCSE {se_sd})");
}

/// Proposal D3: a start exactly on a bound is refused by the methods that
/// move in the transformed coordinate, with an interior value to use; IF2
/// refuses it before any chain runs. `nl-sbplx`, which optimises on the
/// natural scale from the start as given, runs from it.
#[test]
fn a_start_on_a_bound_is_refused_by_the_transformed_samplers_only() {
    if camdlc_bin().is_none() {
        return;
    }
    let tmp = tempdir("edge_refusal");
    let ir = compile(&tmp.0, "edge", &model_src(C_REAL));
    let est = "c = { bounds = [0, 1], start = 0.0 }";
    let nl = "algorithm = \"nl-sbplx\"\nbackend = \"ode\"\nchains = 1\nmax_evals = 20\nstarts = \"single\"\n[method.dt_check]\nenabled = false";
    let if2 = format!("{IF2}\nstarts = \"single\"");
    let refusal = "`c` starts on its lower bound 0";
    let (out, _) = run_fit(&tmp.0, &ir, est, &if2, "if2");
    assert!(!out.status.success(), "if2 must refuse a start on a bound:\n{}", stderr(&out));
    assert!(stderr(&out).contains(refusal) && stderr(&out).contains("e.g. at 0.01"),
        "the refusal names the bound and an interior value:\n{}", stderr(&out));
    let (out, _) = run_fit(&tmp.0, &ir, est, nl, "nl");
    assert!(out.status.success(), "nl-sbplx: {}", stderr(&out));
    assert!(!stderr(&out).contains(refusal), "nl-sbplx uses the start as given:\n{}", stderr(&out));
}
