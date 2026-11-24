//! End-to-end acceptance for `camdl compare` on stored fit handles: each fit
//! run is predicted and scored by compare itself, so no pre-run
//! `pfilter --save-prequential` or `camdl score` is needed. Two fit handles
//! give a comparison table, both fits scored on the same data so their
//! T_score agrees; a sampler's fit is scored on its posterior draws, over the
//! default 5 replicates; and a per-fit chain exclusion moves only the named
//! fit. That a one-step score equals `pfilter`'s log-likelihood at the same
//! point and seed is `camdl score`'s own test.

mod fit_store_support;

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn skip_if_missing_binary() -> PathBuf {
    let bin = binary();
    assert!(
        bin.exists(),
        "release camdl binary missing: {} — run `make build-rust` or `make test`",
        bin.display()
    );
    bin
}

/// A closed SIR with a weekly NegBinomial observation — small and well-behaved.
/// `rho` (reporting rate) is a fixed parameter, so two fits with different `rho`
/// are genuinely distinct (distinct fit hashes) yet score the same data axis.
const MODEL: &str = r#"
time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate         in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate         in [0.01, 0.5]  ~ log_normal(mu = -2.0, sigma = 0.5)
  N0    : count
  I0    : count
  rho   : probability  in [0.05, 0.95] ~ beta(alpha = 2.0, beta = 5.0)
  k     : positive     in [1.0, 100.0] ~ half_normal(sigma = 10.0)
}

let N = S + I + R

transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}

init {
  S = N0 - I0
  I = I0
}

observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    covers        = closing_at(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}

simulate {
  from = 0 'days
  to   = 80 'days
}
"#;

/// A short observed weekly series (rise-and-fall), times on the weekly grid.
const DATA: &str = "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n";

/// A cheap IF2 fit (point estimate). `rho` is set per-fit so two configs get
/// distinct problems, each run under its own `@label`.
fn fit_toml(rho: f64) -> String {
    format!(
        r#"store = "results"

[model]
camdl = "model.camdl"

[data.observations]
weekly_cases = "weekly_cases.tsv"

[estimate]
beta  = {{ bounds = [0.05, 1.0], start = 0.4 }}
gamma = {{ bounds = [0.01, 0.5], start = 0.15 }}

[fixed]
N0  = 10000
I0  = 10
rho = {rho}
k   = 10.0

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 150
iterations = 15
cooling = 0.7
"#
    )
}

fn run(bin: &Path, dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(bin).env_remove("CLICOLOR_FORCE")
        .args(args)
        .current_dir(dir)
        // `@label` handles resolve in the store the fit files name.
        .env("CAMDL_STORE", dir.join("results"))
        // Ad-hoc run: skip the camdlc git-hash handshake (the binary under test
        // is self-consistent). Mirrors the runbook's ad-hoc guidance.
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

/// Stand up a tmp workspace with the model + data, then run the two labeled IF2
/// fits (`@a` with rho=0.5, `@b` with rho=0.6). Returns the tmp dir.
fn setup_two_fits(bin: &Path, slug: &str) -> PathBuf {
    let tmp = std::env::temp_dir().join(format!("camdl_compare_auto_{slug}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), MODEL).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(tmp.join("a.toml"), fit_toml(0.5)).unwrap();
    std::fs::write(tmp.join("b.toml"), fit_toml(0.6)).unwrap();

    for (cfg, label) in [("a.toml", "a"), ("b.toml", "b")] {
        let out = run(bin, &tmp, &["fit", "run", cfg, "--label", label, "--seed", "1"]);
        assert!(
            out.status.success(),
            "fit run {cfg} failed:\nstdout={}\nstderr={}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    tmp
}

/// The folder of the fit run labelled `label` in the store `results`: the
/// `fit_run` result of the last `fit run --label <label>` invocation the store
/// recorded.
fn run_with_label(results: &Path, label: &str) -> PathBuf {
    let mut last: Option<(u64, String)> = None;
    for e in std::fs::read_dir(results.join(".invocations")).unwrap().flatten() {
        let v = fit_store_support::json(&e.path());
        if v["label"] != label {
            continue;
        }
        let seq = v["seq"].as_u64().unwrap();
        let id = v["results"].as_array().unwrap().iter()
            .find(|r| r["kind"] == "fit_run")
            .and_then(|r| r["run_id"].as_str())
            .map(str::to_string);
        if let Some(id) = id {
            if last.as_ref().is_none_or(|(s, _)| seq > *s) {
                last = Some((seq, id));
            }
        }
    }
    let (_, id) = last.unwrap_or_else(|| panic!("no fit run labeled {label} under {}", results.display()));
    fit_store_support::runs(results).into_iter()
        .find(|r| fit_store_support::run_id(r) == id)
        .unwrap_or_else(|| panic!("run {id} labeled {label} is not under {}", results.display()))
}

/// A cheap PGAS fit (a posterior cloud — writes `fit_state.toml` + `draws.tsv`,
/// NO `final_params.toml`). `rho` fixed per-fit for distinct hashes.
fn pgas_fit_toml(rho: f64) -> String {
    format!(
        r#"store = "results"

[model]
camdl = "model.camdl"

[data.observations]
weekly_cases = "weekly_cases.tsv"

[estimate]
beta  = {{ bounds = [0.05, 1.0], start = 0.4 }}
gamma = {{ bounds = [0.01, 0.5], start = 0.15 }}

[fixed]
N0  = 10000
I0  = 10
rho = {rho}
k   = 10.0

[method]
algorithm = "pgas"
backend = "chain_binomial"
chains = 2
particles = 200
sweeps = 60
burn_in = 20
thin = 1
"#
    )
}

#[test]
fn compare_auto_derives_prequential_from_two_pgas_fits() {
    // The headline Bayesian comparison: two PGAS fits, each predicted from
    // its posterior draws.
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir().join(format!("camdl_compare_pgas_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), MODEL).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(tmp.join("a.toml"), pgas_fit_toml(0.5)).unwrap();
    std::fs::write(tmp.join("b.toml"), pgas_fit_toml(0.6)).unwrap();
    for (cfg, label) in [("a.toml", "a"), ("b.toml", "b")] {
        let out = run(&bin, &tmp, &["fit", "run", cfg, "--label", label, "--seed", "1"]);
        assert!(
            out.status.success(),
            "pgas fit run {cfg} failed:\nstderr={}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    // --draws 4 keeps the mixture path exercised without the default-64
    // pass count; the resolution path under test is unchanged.
    let out = run(&bin, &tmp, &["compare", "@a", "@b", "--particles", "300",
        "--seed", "1", "--draws", "4"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "compare @a @b on two PGAS fits must succeed:\nstdout={stdout}\nstderr={stderr}"
    );
    assert!(stdout.contains("elpd"), "comparison table has an elpd column:\n{stdout}");
    assert!(
        stdout.lines().any(|l| l.trim_start().starts_with("@a"))
            && stdout.lines().any(|l| l.trim_start().starts_with("@b")),
        "both PGAS fits appear as rows:\n{stdout}"
    );

    // Stage 4.1 (§3.6): a Bayesian fit's derived predictive is the
    // posterior mixture — provenance says so, with the mixture size.
    let out = run(&bin, &tmp, &["compare", "@a", "@b", "--particles", "300",
        "--seed", "1", "--draws", "3", "--format", "json"]);
    assert!(out.status.success(), "mixture compare failed:\nstderr={}",
        String::from_utf8_lossy(&out.stderr));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    for row in v["rows"].as_array().unwrap() {
        assert_eq!(row["provenance"]["posterior"]["n_draws"],
            serde_json::json!(3),
            "a Bayesian fit's row must be stamped posterior: {row}");
        // The row records what was predicted: 3 draws of the retained cloud,
        // over the default 5 replicates.
        let d = &row["derivation"];
        assert_eq!(d["theta"]["posterior_draws"]["n_used"], 3, "{row}");
        assert!(d["theta"]["posterior_draws"]["n_available"].as_u64().is_some_and(|n| n >= 3),
            "thinned from the whole retained cloud: {row}");
        assert_eq!(d["replicates"], 5, "{row}");
        assert_eq!(d["particles"], 300, "{row}");
        assert_eq!(d["excluded_chains"], serde_json::json!([]), "{row}");
    }

    // Stage 4.2: replicate derives attach a filter-noise MC SE per row and
    // the pair's combined SE; --replicates 1 attaches none.
    let out = run(&bin, &tmp, &["compare", "@a", "@b", "--particles", "300",
        "--seed", "1", "--draws", "1", "--replicates", "2", "--format", "json"]);
    assert!(out.status.success(), "replicate compare failed:\nstderr={}",
        String::from_utf8_lossy(&out.stderr));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    for row in v["rows"].as_array().unwrap() {
        assert_eq!(row["derivation"]["theta"]["posterior_draws"]["n_used"], 1,
            "--draws 1 scores one posterior draw: {row}");
        assert_eq!(row["derivation"]["replicates"], 2, "{row}");
        let mc = row["mc_se_elpd"].as_f64()
            .unwrap_or_else(|| panic!("replicates must attach mc_se_elpd: {row}"));
        assert!(mc.is_finite() && mc >= 0.0, "mc_se_elpd must be finite: {row}");
    }
    let out_table = run(&bin, &tmp, &["compare", "@a", "@b", "--particles", "300",
        "--seed", "1", "--draws", "1", "--replicates", "2"]);
    assert!(out_table.status.success());
    assert!(String::from_utf8_lossy(&out_table.stdout)
        .contains("filter-noise MC SE"),
        "the MC SE note is printed with the table:\n{}",
        String::from_utf8_lossy(&out_table.stdout));

    let out = run(&bin, &tmp, &["compare", "@a", "@b", "--particles", "300",
        "--seed", "1", "--draws", "1", "--replicates", "1", "--format", "json"]);
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(v["rows"][0]["mc_se_elpd"].is_null(),
        "--replicates 1 must attach no MC SE");

    // --draws 1 is one posterior draw, still a posterior; the posterior-mean
    // plug-in is a fixed point (`fit summary --params-only`).
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    for row in v["rows"].as_array().unwrap() {
        assert_eq!(row["provenance"], serde_json::json!({"posterior": {"n_draws": 1}}), "{row}");
    }

    // Provenance preflight: a posterior-mixture row against a plug-in row
    // (here an explicit prequential.json, which records a plug-in trace) is
    // refused; --allow-mixed-provenance lifts exactly that refusal.
    std::fs::write(tmp.join("explicit.json"), r#"{
        "schema_version": 7, "t0": 0, "provenance": "plug_in",
        "conditioning": "in_sample",
        "steps": [{"t": 7.0, "y_obs": 1.0, "y_pred_samples": [],
                   "log_score": -1.0, "crps": 0.5, "pit": 0.5, "ess": 100.0}],
        "warnings": []
    }"#).unwrap();
    let out = run(&bin, &tmp, &["compare", "@a", "explicit.json",
        "--particles", "300", "--seed", "1", "--draws", "3"]);
    assert!(!out.status.success(), "mixed provenance must be refused");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("mix provenance kinds"),
        "must name the mixed provenance:\n{stderr}");
    let out = run(&bin, &tmp, &["compare", "@a", "explicit.json",
        "--particles", "300", "--seed", "1", "--draws", "3",
        "--allow-mixed-provenance"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("mix provenance kinds"),
        "--allow-mixed-provenance must lift the provenance refusal:\n{stderr}");

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn compare_auto_derives_prequential_from_two_fit_handles() {
    // `camdl compare @a @b` of two IF2 runs with nothing stored beforehand:
    // each is predicted at its winning point and scored, and the table renders
    // with both rows and an elpd column, both fits' T_score equal.
    let bin = skip_if_missing_binary();
    let tmp = setup_two_fits(&bin, "table");

    // Low particle count keeps the test fast.
    let out = run(&bin, &tmp, &["compare", "@a", "@b", "--particles", "300", "--seed", "1"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "compare @a @b (auto-derive) must exit 0:\nstdout={stdout}\nstderr={stderr}"
    );

    // The table has an elpd column and a row for each fit handle.
    assert!(stdout.contains("elpd"), "comparison table carries an elpd column:\n{stdout}");
    let row_a = stdout
        .lines()
        .find(|l| l.trim_start().starts_with("@a"))
        .unwrap_or_else(|| panic!("no @a row in the table:\n{stdout}"));
    let row_b = stdout
        .lines()
        .find(|l| l.trim_start().starts_with("@b"))
        .unwrap_or_else(|| panic!("no @b row in the table:\n{stdout}"));

    // Both fits scored the same data at the same particles/seed → identical
    // T_score (column 3 of each row, after the model name and its role).
    let t_a = row_a.split_whitespace().nth(2).expect("@a T_score cell");
    let t_b = row_b.split_whitespace().nth(2).expect("@b T_score cell");
    assert_eq!(
        t_a, t_b,
        "uniform derive settings → commensurable T_score; rows:\n{row_a}\n{row_b}"
    );
    // T_score is a positive integer (a real scored horizon, not 0/blank).
    assert!(
        t_a.parse::<usize>().map(|n| n > 0).unwrap_or(false),
        "T_score is a positive count, got {t_a:?}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

/// Recursively locate the single `draws.tsv` under a fit run folder.
fn find_draws_tsv(dir: &Path) -> Option<PathBuf> {
    for e in std::fs::read_dir(dir).ok()?.flatten() {
        let p = e.path();
        if p.is_dir() {
            if let Some(found) = find_draws_tsv(&p) {
                return Some(found);
            }
        } else if p.file_name().and_then(|n| n.to_str()) == Some("draws.tsv") {
            return Some(p);
        }
    }
    None
}

/// The absolute elpd of the row named `name` from `compare --format json` stdout.
fn elpd_of(json_stdout: &str, name: &str) -> f64 {
    let v: serde_json::Value = serde_json::from_str(json_stdout).unwrap_or_else(|e| {
        panic!("compare --format json did not emit JSON ({e}):\n{json_stdout}")
    });
    v["rows"]
        .as_array()
        .expect("rows array")
        .iter()
        .find(|r| r["name"] == name)
        .unwrap_or_else(|| panic!("no row named {name} in:\n{json_stdout}"))["elpd"]
        .as_f64()
        .expect("elpd is a finite number")
}

#[test]
fn compare_per_fit_exclude_chains_rescores_only_the_named_fit() {
    // gh#417/gh#418: `--exclude-chains @a:1` must drop chain 1 from @a's posterior
    // before @a is predicted — the subset `predict` bands — while leaving every
    // other fit whole. Excluding a chain moves @a's draws (hence its elpd), and
    // @b (unnamed) must be identical to the all-chains run. If the token bound
    // cohort-wide, @b would move too; if it were ignored, @a would not move.
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir().join(format!("camdl_compare_excl_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), MODEL).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(tmp.join("a.toml"), pgas_fit_toml(0.5)).unwrap();
    std::fs::write(tmp.join("b.toml"), pgas_fit_toml(0.6)).unwrap();
    for (cfg, label) in [("a.toml", "a"), ("b.toml", "b")] {
        let out = run(&bin, &tmp, &["fit", "run", cfg, "--label", label, "--seed", "1"]);
        assert!(
            out.status.success(),
            "pgas fit run {cfg} failed:\nstderr={}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    // The draws of @a's chain 0, which are what remains once chain 1 is
    // excluded.
    let run_a = run_with_label(&tmp.join("results"), "a");
    let draws_a = std::fs::read_to_string(find_draws_tsv(&run_a).expect("@a wrote a draws.tsv"))
        .unwrap();
    let mut draw_lines = draws_a.lines().filter(|l| !l.starts_with('#'));
    let chain_col = draw_lines.next().unwrap().split('\t').position(|c| c == "chain").unwrap();
    let chain0_draws = draw_lines.filter(|l| l.split('\t').nth(chain_col) == Some("0")).count();
    assert!(chain0_draws > 0, "@a's chain 0 holds draws:\n{draws_a}");

    // Baseline pinned to @b so neither row's absolute elpd depends on which is
    // baseline; JSON reports each row's own elpd.
    let common = ["compare", "@a", "@b", "--baseline", "@b", "--format", "json",
                  "--particles", "300", "--seed", "1", "--draws", "4", "--replicates", "1"];
    let all = run(&bin, &tmp, &common);
    assert!(
        all.status.success(),
        "compare (all chains) failed:\nstderr={}",
        String::from_utf8_lossy(&all.stderr)
    );
    let all_out = String::from_utf8_lossy(&all.stdout);
    let (e_a_all, e_b_all) = (elpd_of(&all_out, "@a"), elpd_of(&all_out, "@b"));

    // Per-fit: drop @a's chain 1, leave @b untouched.
    let mut sub_args = common.to_vec();
    sub_args.extend_from_slice(&["--exclude-chains", "@a:1"]);
    let sub = run(&bin, &tmp, &sub_args);
    assert!(
        sub.status.success(),
        "compare --exclude-chains @a:1 failed:\nstderr={}",
        String::from_utf8_lossy(&sub.stderr)
    );
    let sub_out = String::from_utf8_lossy(&sub.stdout);
    let (e_a_sub, e_b_sub) = (elpd_of(&sub_out, "@a"), elpd_of(&sub_out, "@b"));
    // The row records the exclusion and the cloud it left: the retained draws
    // of chain 0.
    let v: serde_json::Value = serde_json::from_str(&sub_out).unwrap();
    let row = |name: &str| v["rows"].as_array().unwrap().iter()
        .find(|r| r["name"] == name).unwrap().clone();
    assert_eq!(row("@a")["derivation"]["excluded_chains"], serde_json::json!([1]),
        "@a's trace records the dropped chain: {v}");
    assert_eq!(row("@a")["derivation"]["theta"],
        serde_json::json!({"posterior_draws": {"n_used": 4, "n_available": chain0_draws}}), "{v}");
    assert_eq!(row("@b")["derivation"]["excluded_chains"], serde_json::json!([]),
        "@b dropped nothing: {v}");

    // @a's draws come from chain 0 alone → its elpd changes.
    assert!(
        (e_a_all - e_a_sub).abs() > 1e-6,
        "excluding @a's chain 1 must change @a's elpd (draws from the retained \
         subset): all={e_a_all}, subset={e_a_sub}"
    );
    // @b was not named → byte-identical: the token bound per-fit, not cohort-wide.
    assert!(
        (e_b_all - e_b_sub).abs() < 1e-9,
        "@b was not named in --exclude-chains @a:1, so its elpd must be unchanged \
         (per-fit binding, not cohort-wide): all={e_b_all}, subset={e_b_sub}"
    );

    // The bias warning fired and named the targeted fit.
    let sub_err = String::from_utf8_lossy(&sub.stderr);
    assert!(
        sub_err.contains("--exclude-chains") && sub_err.contains("fit '@a'"),
        "compare --exclude-chains @a:1 must warn, naming fit @a:\nstderr={sub_err}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}
