//! End-to-end acceptance for `camdl compare` auto-deriving a prequential trace
//! from a sealed fit handle (Phase 2a) — no pre-run `pfilter --save-prequential`
//! needed. Two tests:
//!
//!  (a) two fit handles → a comparison table, both fits scored on the same data
//!      so their T_score agrees;
//!  (b) the CORRECTNESS GATE: the trace `compare` derives from a handle is
//!      numerically identical to a hand-run `camdl pfilter` at the same θ̂ /
//!      particles / seed — proof the derive path reuses the canonical filter
//!      rather than reimplementing it.
//!
//! Phase 2a of
//! docs/dev/proposals/2026-06-27-sealed-fit-packets-handles-and-override-algebra.md
//! (handle-aware `compare`; the prequential scoring is
//! docs/dev/proposals/2026-04-20-prequential-evaluation.md §8).

use sim::inference::prequential::PrequentialTrace;
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
/// distinct fit hashes → distinct segments → distinct `@label`s.
fn fit_toml(rho: f64) -> String {
    format!(
        r#"output_dir = "results"

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

[stages.scout]
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
    Command::new(bin)
        .args(args)
        .current_dir(dir)
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

/// Find the fit segment dir whose sidecar carries `label`.
fn segment_with_label(results: &Path, label: &str) -> PathBuf {
    let fits = results.join("fits");
    for e in std::fs::read_dir(&fits).unwrap().flatten() {
        let meta = e.path().join("fit.meta.json");
        if let Ok(txt) = std::fs::read_to_string(&meta) {
            if let Ok(j) = serde_json::from_str::<serde_json::Value>(&txt) {
                if j["label"] == label {
                    return e.path();
                }
            }
        }
    }
    panic!("no fit segment labeled {label} under {}", fits.display());
}

/// A cheap PGAS fit (a posterior cloud — writes `fit_state.toml` + `draws.tsv`,
/// NO `final_params.toml`). `rho` fixed per-fit for distinct hashes.
fn pgas_fit_toml(rho: f64) -> String {
    format!(
        r#"output_dir = "results"

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

[stages.posterior]
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
    // gh#322 review (Blocker 2): the HEADLINE Bayesian comparison. A PGAS fit
    // writes `fit_state.toml` + `draws.tsv` but NO `final_params.toml`, so the
    // old derive (winner_params_toml → final_params.toml) dead-ended
    // file-not-found — `compare @pgas_a @pgas_b`, the whole point of Phase 2a,
    // failed. Routed through the draws-cloud authority (θ̂ = posterior mean over
    // draws.tsv), it succeeds.
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
        "compare @a @b on two PGAS fits must succeed (θ̂ = posterior mean, not a \
         missing final_params.toml):\nstdout={stdout}\nstderr={stderr}"
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
    }

    // --draws 1 is the documented cheap mode: plug-in at the posterior mean.
    let out = run(&bin, &tmp, &["compare", "@a", "@b", "--particles", "300",
        "--seed", "1", "--draws", "1", "--format", "json"]);
    assert!(out.status.success(), "--draws 1 compare failed:\nstderr={}",
        String::from_utf8_lossy(&out.stderr));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    for row in v["rows"].as_array().unwrap() {
        assert_eq!(row["provenance"], serde_json::json!("plug_in"),
            "--draws 1 must fall back to the plug-in point: {row}");
    }

    // Provenance preflight: a posterior-mixture row against a plug-in row
    // (here an explicit prequential.json, which records a plug-in trace) is
    // refused; --allow-mixed-provenance lifts exactly that refusal.
    std::fs::write(tmp.join("explicit.json"), r#"{
        "schema_version": 3, "t0": 0, "provenance": "plug_in",
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
    // (a) `camdl compare @a @b` with NO pre-existing prequential.json: both
    // traces are auto-derived at θ̂ via the canonical pfilter, and the table
    // renders with both rows and an elpd column, both fits' T_score equal.
    let bin = skip_if_missing_binary();
    let tmp = setup_two_fits(&bin, "table");

    // Low particle count keeps the test fast; correctness is gated by test (b).
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
    // T_score (column 2 of each row, after the left-aligned model name).
    let t_a = row_a.split_whitespace().nth(1).expect("@a T_score cell");
    let t_b = row_b.split_whitespace().nth(1).expect("@b T_score cell");
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

#[test]
fn compare_derived_trace_equals_manual_pfilter() {
    // (b) CORRECTNESS GATE. For fit `@a`, the trace `compare` derives must be
    // numerically identical to a hand-run `camdl pfilter` at the same θ̂ /
    // particles / seed — proof the derive path routes through the one canonical
    // filter rather than a divergent reimplementation.
    let bin = skip_if_missing_binary();
    let tmp = setup_two_fits(&bin, "gate");
    let results = tmp.join("results");

    const PARTICLES: &str = "500";
    const SEED: &str = "7";

    // θ̂ for @a — exactly what `winner_params_toml` feeds the derive path.
    let out = run(&bin, &tmp, &["fit", "summary", "@a", "--params-only"]);
    assert!(
        out.status.success(),
        "fit summary @a --params-only failed:\nstderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    std::fs::write(tmp.join("theta.toml"), &out.stdout).unwrap();

    // Manual: run the canonical pfilter at θ̂ on @a's ARCHIVED model IR (the
    // same model the derive path uses), at the same particles + seed.
    let model_a = segment_with_label(&results, "a").join("model.ir.json");
    assert!(model_a.is_file(), "archived model.ir.json must exist for @a");
    let data_abs = tmp.join("weekly_cases.tsv");
    let data_arg = format!("weekly_cases={}", data_abs.display());
    let out = run(
        &bin,
        &tmp,
        &[
            "pfilter",
            &model_a.to_string_lossy(),
            "--data",
            &data_arg,
            "--params",
            "theta.toml",
            "--save-prequential",
            "manual",
            "--particles",
            PARTICLES,
            "--seed",
            SEED,
        ],
    );
    assert!(
        out.status.success(),
        "manual pfilter failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let manual: PrequentialTrace =
        serde_json::from_str(&std::fs::read_to_string(tmp.join("manual.json")).unwrap())
            .expect("manual prequential.json parses");

    // Auto: compare derives @a's trace at the SAME particles + seed. Read the
    // JSON and pull @a's row.
    let out = run(
        &bin,
        &tmp,
        &["compare", "@a", "@b", "--particles", PARTICLES, "--seed", SEED, "--format", "json"],
    );
    assert!(
        out.status.success(),
        "compare --format json failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("compare --format json emits valid JSON");
    let rows = v["rows"].as_array().expect("rows array");
    let row_a = rows
        .iter()
        .find(|r| r["path"] == "@a")
        .expect("@a row in compare JSON");
    let derived_t = row_a["t_score"].as_u64().expect("t_score is an integer") as usize;
    let derived_elpd = row_a["elpd"].as_f64().expect("elpd is a number");

    // Same code path, same inputs, same seed ⇒ identical. T_score / n_scored
    // exact; elpd bit-identical (tight tolerance flags any real divergence).
    assert_eq!(
        derived_t,
        manual.n_scored(),
        "derived T_score must equal the manual pfilter's n_scored"
    );
    assert!(
        (derived_elpd - manual.elpd()).abs() < 1e-9,
        "derived elpd must equal manual elpd (same θ̂/particles/seed via the same \
         canonical filter): derived={derived_elpd} manual={}",
        manual.elpd()
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

/// Recursively locate the single `draws.tsv` under a fit segment.
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
    // gh#417/gh#418: `--exclude-chains @a:2` must drop chain 2 from @a's posterior
    // BEFORE deriving its plug-in θ̂ — scoring the SAME subset a `predict` would
    // band (predict/summary already do — gh#409) — while leaving every OTHER fit
    // whole. Inject a stuck minority chain into @a's cloud: excluding it moves
    // @a's θ̂ (hence prequential elpd), and @b (unnamed) must be byte-identical to
    // the all-chains run. If the token bound cohort-wide, @b would move too; if it
    // were ignored, @a would not move.
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

    // Overwrite @a's cloud with two chains: chain 1 (0-based 0) mixed at
    // beta≈0.35; chain 2 (0-based 1) stuck at beta≈0.90 (a fast epidemic).
    // Columns are this model's draws.tsv order — estimated (beta, gamma) then
    // fixed (N0, I0, rho, k); the keyed loader maps by name, so what matters is
    // that the stuck chain carries a materially different beta.
    let seg_a = segment_with_label(&tmp.join("results"), "a");
    let draws_a = find_draws_tsv(&seg_a).expect("@a wrote a draws.tsv");
    let mut cloud = String::from("chain\tdraw\tbeta\tgamma\tN0\tI0\trho\tk\n");
    for draw in 0..6 {
        cloud.push_str(&format!("0\t{draw}\t0.35\t0.14\t10000\t10\t0.5\t10\n"));
    }
    for draw in 0..6 {
        cloud.push_str(&format!("1\t{draw}\t0.90\t0.05\t10000\t10\t0.5\t10\n"));
    }
    std::fs::write(&draws_a, cloud).unwrap();

    // Baseline pinned to @b so neither row's absolute elpd depends on which is
    // baseline; JSON reports each row's own elpd. `--draws 1` (plug-in):
    // this test is about chain EXCLUSION moving θ̂; under the mixture
    // default, the deliberately stuck chain's draws are mixture components
    // whose filter degenerates, which is its own (tested) refusal.
    let common = ["compare", "@a", "@b", "--baseline", "@b", "--format", "json",
                  "--particles", "300", "--seed", "1", "--draws", "1"];
    let all = run(&bin, &tmp, &common);
    assert!(
        all.status.success(),
        "compare (all chains) failed:\nstderr={}",
        String::from_utf8_lossy(&all.stderr)
    );
    let all_out = String::from_utf8_lossy(&all.stdout);
    let (e_a_all, e_b_all) = (elpd_of(&all_out, "@a"), elpd_of(&all_out, "@b"));

    // Per-fit: drop @a's stuck chain, leave @b untouched.
    let mut sub_args = common.to_vec();
    sub_args.extend_from_slice(&["--exclude-chains", "@a:2"]);
    let sub = run(&bin, &tmp, &sub_args);
    assert!(
        sub.status.success(),
        "compare --exclude-chains @a:2 failed:\nstderr={}",
        String::from_utf8_lossy(&sub.stderr)
    );
    let sub_out = String::from_utf8_lossy(&sub.stdout);
    let (e_a_sub, e_b_sub) = (elpd_of(&sub_out, "@a"), elpd_of(&sub_out, "@b"));

    // @a's θ̂ moves off the fast-epidemic pull → its elpd changes.
    assert!(
        (e_a_all - e_a_sub).abs() > 1e-6,
        "excluding @a's stuck chain must change @a's elpd (θ̂ over the retained \
         subset): all={e_a_all}, subset={e_a_sub}"
    );
    // @b was not named → byte-identical: the token bound per-fit, not cohort-wide.
    assert!(
        (e_b_all - e_b_sub).abs() < 1e-9,
        "@b was not named in --exclude-chains @a:2, so its elpd must be unchanged \
         (per-fit binding, not cohort-wide): all={e_b_all}, subset={e_b_sub}"
    );

    // The bias warning fired and named the targeted fit.
    let sub_err = String::from_utf8_lossy(&sub.stderr);
    assert!(
        sub_err.contains("--exclude-chains") && sub_err.contains("fit '@a'"),
        "compare --exclude-chains @a:2 must warn, naming fit @a:\nstderr={sub_err}"
    );

    // Stage 4.1: under the MIXTURE, the stuck chain's draws are components
    // whose filter degenerates — refused with guidance, not repaired
    // (dropping degenerate components would bias the mixture toward
    // well-behaved θ). Excluding the sick chain makes the mixture derivable.
    let mixed = run(&bin, &tmp, &["compare", "@a", "@b", "--baseline", "@b",
        "--format", "json", "--particles", "300", "--seed", "1", "--draws", "4"]);
    assert!(!mixed.status.success(),
        "a degenerate mixture component must fail the derive");
    let mixed_err = String::from_utf8_lossy(&mixed.stderr);
    assert!(
        mixed_err.contains("mixture component") && mixed_err.contains("--exclude-chains"),
        "the refusal names the component and the fix:\n{mixed_err}"
    );
    let healthy = run(&bin, &tmp, &["compare", "@a", "@b", "--baseline", "@b",
        "--format", "json", "--particles", "300", "--seed", "1", "--draws", "4",
        "--exclude-chains", "@a:2"]);
    assert!(healthy.status.success(),
        "dropping the sick chain must make the mixture derivable:\nstderr={}",
        String::from_utf8_lossy(&healthy.stderr));

    let _ = std::fs::remove_dir_all(&tmp);
}

/// gh#634: a fit toml that declares `condition_from` must have it FORWARDED
/// to the pfilter `compare` derives the prequential with — otherwise the
/// derived scores cover a window the fit never scored, and pfilter's
/// wide-first-window guard (W329) refuses with advice to set condition_from,
/// which the fit toml already sets (the wrong layer). The data's first obs is
/// pushed late so the unconditioned child pfilter would hard-error: pre-fix,
/// this compare FAILED with that misdirected message; post-fix it derives.
#[test]
fn compare_forwards_condition_from_to_derived_pfilter() {
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir()
        .join(format!("camdl_cmp_cond_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), MODEL).unwrap();
    // First obs at t = 42 on a 7-day cadence: ratio 6 > the W329 threshold.
    std::fs::write(tmp.join("weekly_cases.tsv"),
        "time\tweekly_cases\n42\t626\n49\t327\n56\t91\n63\t45\n").unwrap();
    let toml = fit_toml(0.6).replace(
        "output_dir = \"results\"",
        "output_dir = \"results\"\ncondition_from = \"first_obs - 1 week\"");
    std::fs::write(tmp.join("fit.toml"), &toml).unwrap();

    let out = run(&bin, &tmp, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(out.status.success(), "fit run failed:\nstderr={}",
        String::from_utf8_lossy(&out.stderr));

    let out = run(&bin, &tmp, &["compare", "fit.toml", "fit.toml"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(),
        "compare must forward the fit's condition_from to its derived \
         pfilter (gh#634):\nstdout={stdout}\nstderr={stderr}");
    assert!(!stderr.contains("State the conditioning window"),
        "the W329 misdirection must not fire:\n{stderr}");
}
