//! gh#147 (M3.2) — content-addressed fit-stage acceptance gates.
//!
//! Two properties the fit-stage CAS migration must guarantee, exercised
//! end-to-end against the release `camdl` binary:
//!
//! 1. **Chained-stage reuse** — editing a downstream (posterior) stage's
//!    config must re-key ONLY that stage; the upstream (scout) leaf is a
//!    cache hit and keeps its `run_id`. This is the deps-DAG / `FitDigest`-
//!    excludes-`[stages.*]` factoring working: the expensive scout is cached
//!    across posterior-config iteration.
//! 2. **`--parallel` determinism** — the same fit at 1 vs 8 rayon threads
//!    must produce a bit-identical θ̂. CAS fits run watchdog-None
//!    (machine-speed-independent) and the engine is parallel-invariant, so
//!    the fit is a pure function of its inputs.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The release binary; skip (pass vacuously) when it isn't built — the gate
/// runner builds `--release` first, so a skip in plain `cargo test` is fine.
fn bin() -> PathBuf {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/release/camdl");
    assert!(
        p.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test` (gh#105)",
        p.display()
    );
    p
}

fn model_ir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../sim/tests/fixtures/seed_timing_dated.ir.json")
}

fn write_data(dir: &Path) -> PathBuf {
    let body = "time\tcases\n\
        2020-03-15\t3\n2020-03-16\t6\n2020-03-17\t11\n2020-03-18\t18\n\
        2020-03-19\t27\n2020-03-20\t31\n2020-03-21\t28\n2020-03-22\t20\n\
        2020-03-23\t13\n2020-03-24\t8\n";
    let p = dir.join("cases.tsv");
    std::fs::write(&p, body).unwrap();
    p
}

/// A two-stage fit.toml: `scout` (if2) → `posterior` (if2, `init_mle =
/// "scout"`). `posterior_iters` parameterizes the posterior's config so the
/// reuse test can edit it.
fn write_fit_toml(dir: &Path, out: &Path, data: &Path, posterior_iters: u32) -> PathBuf {
    let body = format!(
        r#"output_dir = "{out}"
# seed_timing_dated starts ~20 days before the first datum (daily incidence) to
# estimate the seed time, so the first incidence bin would otherwise span the
# whole 20-day gap (gh#134 / W329). Condition one cadence before the data: the
# pre-data span is a warm-up, the first datum is scored against one day.
condition_from = "first_obs - 1 day"

[model]
camdl = "{ir}"

[data.observations]
cases = "{data}"

[estimate]
beta = {{ bounds = [0.1, 2.0], start = 0.6 }}
tau  = {{ bounds = [0.0, 60.0], start = 20.0 }}

[fixed]
gamma = 0.2
lambda = 2.0
w = 3.0
N0 = 1000
rho = 0.6
k = 10.0

[stages.scout]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 300
iterations = 4
cooling = 0.7

[stages.posterior]
algorithm = "if2"
backend = "chain_binomial"
init_mle = "scout"
chains = 2
particles = 300
iterations = {posterior_iters}
cooling = 0.7
"#,
        out = out.display(),
        ir = model_ir().display(),
        data = data.display(),
    );
    let p = dir.join("fit.toml");
    std::fs::write(&p, body).unwrap();
    p
}

fn run_fit(bin: &Path, fit_toml: &Path, threads: usize) -> std::process::Output {
    Command::new(bin)
        .arg("fit").arg("run").arg(fit_toml)
        // The tiny 4-iteration test fit does not converge `tau`; the
        // scout-convergence gate is orthogonal to the CAS properties under
        // test, so bypass it and let the posterior stage run.
        .arg("--allow-nonconverged-scout")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("RAYON_NUM_THREADS", threads.to_string())
        .output()
        .expect("camdl fit run must spawn")
}

/// Like `run_fit` but caps the pool via the `--parallel` FLAG (gh#162) rather
/// than the RAYON_NUM_THREADS env — exercises the flag's wiring end to end.
fn run_fit_parallel(bin: &Path, fit_toml: &Path, parallel: usize) -> std::process::Output {
    Command::new(bin)
        .arg("fit").arg("run").arg(fit_toml)
        .arg("--allow-nonconverged-scout")
        .arg("--parallel").arg(parallel.to_string())
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("camdl fit run must spawn")
}

/// Every CAS fit-stage leaf under `out/fits/`: (stage label, run_id), read
/// from each `run.json` (kind = `fit_stage`).
fn stage_leaves(out: &Path) -> Vec<(String, String)> {
    let mut found = Vec::new();
    let mut stack = vec![out.join("fits")];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == "run.json") {
                let Ok(txt) = std::fs::read_to_string(&p) else { continue };
                let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) else { continue };
                if v.get("kind").and_then(|k| k.as_str()) != Some("fit_stage") {
                    continue;
                }
                let run_id = v["run_id"].as_str().unwrap_or("").to_string();
                let stage = v["levels"].as_array().into_iter().flatten()
                    .find(|l| l.get("name").and_then(|n| n.as_str()) == Some("stage"))
                    .and_then(|l| l.get("label").and_then(|x| x.as_str()))
                    .unwrap_or("")
                    .to_string();
                found.push((stage, run_id));
            }
        }
    }
    found
}

fn stage_run_id(leaves: &[(String, String)], stage_substr: &str) -> String {
    leaves.iter()
        .find(|(label, _)| label.contains(stage_substr))
        .unwrap_or_else(|| panic!("no stage leaf matching '{stage_substr}' in {leaves:?}"))
        .1
        .clone()
}

/// Read the scout stage's θ̂ from its `mle_params.toml`, stripping the
/// `[provenance]` block (a wall-clock timestamp + a fit_hash of the
/// path-differing fit.toml, both of which legitimately differ between runs).
/// Used to assert parallel-invariance of the estimate.
fn read_scout_mle(out: &Path) -> String {
    let leaves = stage_leaves(out);
    let scout_dir = {
        let mut stack = vec![out.join("fits")];
        let mut hit = None;
        while let Some(d) = stack.pop() {
            if d.join("mle_params.toml").exists()
                && d.to_string_lossy().contains("scout")
            {
                hit = Some(d);
                break;
            }
            if let Ok(es) = std::fs::read_dir(&d) {
                for e in es.flatten() { if e.path().is_dir() { stack.push(e.path()); } }
            }
        }
        hit.unwrap_or_else(|| panic!("no scout mle_params.toml in {leaves:?}"))
    };
    let full = std::fs::read_to_string(scout_dir.join("mle_params.toml")).unwrap();
    full.split("[provenance]").next().unwrap_or(&full).trim().to_string()
}

/// Property 1: edit the posterior stage's config → the scout leaf is a cache
/// hit (its `run_id` is unchanged) and only the posterior re-keys.
#[test]
fn chained_stage_reuse_only_rekeys_posterior() {
    let bin = bin();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("out");
    let data = write_data(tmp.path());

    // Run 1: scout + posterior(iterations = 4).
    let toml1 = write_fit_toml(tmp.path(), &out, &data, 4);
    let r1 = run_fit(&bin, &toml1, 1);
    assert!(r1.status.success(), "fit run 1 failed: {}", String::from_utf8_lossy(&r1.stderr));
    let leaves1 = stage_leaves(&out);
    let scout1 = stage_run_id(&leaves1, "scout");
    let post1 = stage_run_id(&leaves1, "posterior");

    // Run 2: edit ONLY the posterior's config (iterations 4 → 8).
    let toml2 = write_fit_toml(tmp.path(), &out, &data, 8);
    let r2 = run_fit(&bin, &toml2, 1);
    assert!(r2.status.success(), "fit run 2 failed: {}", String::from_utf8_lossy(&r2.stderr));
    let stderr2 = String::from_utf8_lossy(&r2.stderr);
    let leaves2 = stage_leaves(&out);

    // CAS keeps *both* posterior leaves after run 2 (iter=4 from run 1 + iter=8
    // from run 2), so assert on the leaf *sets*, not a single non-deterministic
    // `stage_run_id` pick.
    let scouts: std::collections::BTreeSet<&str> = leaves2
        .iter()
        .filter(|(label, _)| label.contains("scout"))
        .map(|(_, id)| id.as_str())
        .collect();
    let posteriors: std::collections::BTreeSet<&str> = leaves2
        .iter()
        .filter(|(label, _)| label.contains("posterior"))
        .map(|(_, id)| id.as_str())
        .collect();

    // The scout is cached: exactly one scout leaf, its run_id unchanged. The
    // factoring (FitDigest excludes [stages.*]; the deps-DAG carries cross-stage
    // invalidation) means a posterior-config edit cannot touch the scout.
    assert_eq!(scouts.len(), 1,
        "scout must be cached (one leaf), got {:?}", scouts);
    assert!(scouts.contains(scout1.as_str()),
        "editing the posterior must NOT re-key the scout (scout cached across \
         posterior iteration); scout1={scout1} scouts={scouts:?}");
    // ...and run 2 served the scout from cache, not recomputed.
    assert!(stderr2.to_lowercase().contains("cache hit"),
        "run 2 must report a cache hit for the unchanged scout; stderr:\n{stderr2}");
    // The posterior re-keys (its config changed → new run_id → new leaf): run
    // 1's iter=4 leaf survives and run 2's iter=8 edit adds a distinct second.
    assert!(posteriors.contains(post1.as_str()),
        "run 1's posterior leaf must survive; post1={post1} posteriors={posteriors:?}");
    assert_eq!(posteriors.len(), 2,
        "editing the posterior (iters 4→8) must produce a distinct second \
         posterior leaf; got {:?}", posteriors);
}

/// Property 2: the same fit at 1 vs 8 rayon threads yields bit-identical θ̂
/// (CAS fits run watchdog-None; the engine is parallel-invariant).
#[test]
fn fit_theta_hat_identical_across_parallelism() {
    let bin = bin();
    let tmp = tempfile::tempdir().unwrap();
    let data = write_data(tmp.path());

    let read_mle = read_scout_mle;

    let out1 = tmp.path().join("out1");
    let out8 = tmp.path().join("out8");
    let toml1 = write_fit_toml(tmp.path(), &out1, &data, 4);
    let toml8 = {
        // Same config, different output dir.
        let p = tmp.path().join("fit8.toml");
        let body = std::fs::read_to_string(&toml1).unwrap()
            .replace(&out1.display().to_string(), &out8.display().to_string());
        std::fs::write(&p, body).unwrap();
        p
    };
    let r1 = run_fit(&bin, &toml1, 1);
    assert!(r1.status.success(), "fit @1 thread failed: {}", String::from_utf8_lossy(&r1.stderr));
    let r8 = run_fit(&bin, &toml8, 8);
    assert!(r8.status.success(), "fit @8 threads failed: {}", String::from_utf8_lossy(&r8.stderr));

    assert_eq!(read_mle(&out1), read_mle(&out8),
        "θ̂ (scout mle_params.toml) must be bit-identical at --parallel 1 vs 8 \
         (CAS fits are watchdog-None and the engine is parallel-invariant)");
}

/// gh#162: the `--parallel` FLAG itself caps the pool (not just the
/// RAYON_NUM_THREADS env), and θ̂ is bit-identical at `--parallel 1` vs
/// `--parallel 4` — confirming the flag is a thread budget, never a numerical
/// knob. (Property 2 covers the env path; this covers the flag path.)
#[test]
fn fit_theta_hat_identical_across_parallel_flag() {
    let bin = bin();
    let tmp = tempfile::tempdir().unwrap();
    let data = write_data(tmp.path());

    let out1 = tmp.path().join("p1");
    let out4 = tmp.path().join("p4");
    let toml1 = write_fit_toml(tmp.path(), &out1, &data, 4);
    let toml4 = {
        // Same config, different output dir (so both runs compute independently).
        let p = tmp.path().join("fit_p4.toml");
        let body = std::fs::read_to_string(&toml1).unwrap()
            .replace(&out1.display().to_string(), &out4.display().to_string());
        std::fs::write(&p, body).unwrap();
        p
    };
    let r1 = run_fit_parallel(&bin, &toml1, 1);
    assert!(r1.status.success(), "fit --parallel 1 failed: {}", String::from_utf8_lossy(&r1.stderr));
    let r4 = run_fit_parallel(&bin, &toml4, 4);
    assert!(r4.status.success(), "fit --parallel 4 failed: {}", String::from_utf8_lossy(&r4.stderr));

    assert_eq!(read_scout_mle(&out1), read_scout_mle(&out4),
        "θ̂ must be bit-identical at --parallel 1 vs --parallel 4 (the flag is a \
         thread budget, not a numerical knob)");
}

/// Property 3 (Q2): `fit run` must announce the directory its stage leaves
/// actually land in. The leaves are written by `resolve_fit_stage` under the
/// `FitDigest` fit-level segment (`fits/{stem}-{h8}/`); the announced
/// `output:` line must be exactly that segment — i.e. the announced path
/// EXISTS and is the parent of the `NN-stage-{h8}` leaf dirs.
///
/// Pre-fix, the announcement used `config.fit_dir()` (a `fit_content_hash`-
/// keyed dir) while the leaves landed under the divergent `FitDigest` dir, so
/// the announced path was empty / nonexistent. This pins them to one basis.
#[test]
fn fit_run_announces_the_real_leaf_directory() {
    let bin = bin();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("out");
    let data = write_data(tmp.path());

    let toml = write_fit_toml(tmp.path(), &out, &data, 4);
    let r = run_fit(&bin, &toml, 1);
    assert!(r.status.success(), "fit run failed: {}", String::from_utf8_lossy(&r.stderr));
    let stderr = String::from_utf8_lossy(&r.stderr);

    // Parse the announced `output:` directory from the startup block.
    let announced = stderr
        .lines()
        .find_map(|l| l.trim_start().strip_prefix("output:"))
        .map(|s| PathBuf::from(s.trim()))
        .unwrap_or_else(|| panic!("no `output:` line in fit run stderr:\n{stderr}"));

    // The announced dir must exist (pre-fix it was an empty/absent
    // fit_content_hash directory while the leaves went elsewhere).
    assert!(
        announced.is_dir(),
        "announced output dir must exist on disk: {}\nstderr:\n{stderr}",
        announced.display(),
    );

    // Locate the actual stage leaf (the dir holding a fit_stage run.json) and
    // confirm the announced dir is its grandparent — the fit segment that
    // `store_path` factors as fits/{fit}/{NN-stage}/{seed}/.
    let leaf_dir = {
        let mut stack = vec![out.join("fits")];
        let mut hit = None;
        while let Some(d) = stack.pop() {
            if d.join("run.json").exists() {
                let txt = std::fs::read_to_string(d.join("run.json")).unwrap_or_default();
                if txt.contains("\"fit_stage\"") {
                    hit = Some(d.clone());
                    break;
                }
            }
            if let Ok(es) = std::fs::read_dir(&d) {
                for e in es.flatten() {
                    if e.path().is_dir() {
                        stack.push(e.path());
                    }
                }
            }
        }
        hit.unwrap_or_else(|| panic!("no fit_stage leaf under {}", out.join("fits").display()))
    };

    // Leaf factoring: fits/{fit}/{NN-stage}/{seed}/run.json — so the fit
    // segment is the seed dir's grandparent (= leaf_dir.parent().parent()).
    let fit_segment = leaf_dir
        .parent()
        .and_then(|p| p.parent())
        .unwrap_or_else(|| panic!("leaf {} has no fit-segment grandparent", leaf_dir.display()));

    assert_eq!(
        announced.canonicalize().unwrap(),
        fit_segment.canonicalize().unwrap(),
        "the announced `output:` dir ({}) must equal the actual fit segment \
         that holds the NN-stage leaf dirs ({})",
        announced.display(),
        fit_segment.display(),
    );

    // And the segment really contains the stage tree (a NN-stage-* child).
    let has_stage_child = std::fs::read_dir(&fit_segment)
        .unwrap()
        .flatten()
        .any(|e| {
            e.path().is_dir()
                && e.file_name().to_string_lossy().contains("-stage-")
                || e.file_name().to_string_lossy().contains("scout")
                || e.file_name().to_string_lossy().contains("posterior")
        });
    assert!(
        has_stage_child,
        "announced fit segment {} must contain the NN-stage leaf dirs",
        fit_segment.display(),
    );
}
