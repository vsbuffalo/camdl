//! Content-addressed fit-run acceptance gates, exercised end-to-end against
//! the release `camdl` binary:
//!
//! 1. **Chained-method reuse**: a second file that warm-starts from the
//!    first (`starts = { from_mle = <run> }`) is stored under the same fit
//!    problem, because the problem's id hashes the problem alone and not
//!    `[method]`; editing the second file's `[method]` adds only a new run,
//!    and re-running the first file is a cache hit that keeps its run id. The
//!    expensive upstream fit is cached across downstream-config iteration.
//! 2. **`--parallel` determinism**: the same fit at 1 vs 8 rayon threads
//!    produces a bit-identical θ̂. Stored fits run without a watchdog
//!    (machine-speed-independent) and the engine is parallel-invariant, so
//!    the fit is a pure function of its inputs.

mod fit_store_support;

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

/// The problem half both files share.
fn problem_half(out: &Path, data: &Path) -> String {
    format!(
        r#"store = "{out}"

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
"#,
        out = out.display(),
        ir = model_ir().display(),
        data = data.display(),
    )
}

/// The upstream fit: a cheap 4-iteration IF2.
fn write_scout_toml(dir: &Path, out: &Path, data: &Path) -> PathBuf {
    let body = format!(
        "{}\n[method]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\n\
         chains = 2\nparticles = 300\niterations = 4\ncooling = 0.7\n",
        problem_half(out, data)
    );
    let p = dir.join("scout.toml");
    std::fs::write(&p, body).unwrap();
    p
}

/// The downstream fit: the same problem, every chain started at `source`'s
/// point estimate. `iters` parameterizes its config so the reuse test can
/// edit it.
fn write_posterior_toml(dir: &Path, out: &Path, data: &Path, source: &Path, iters: u32) -> PathBuf {
    let body = format!(
        "{}\n[method]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\n\
         chains = 2\nparticles = 300\niterations = {iters}\ncooling = 0.7\n\
         starts = {{ from_mle = \"{}\" }}\n",
        problem_half(out, data),
        source.display()
    );
    let p = dir.join("posterior.toml");
    std::fs::write(&p, body).unwrap();
    p
}

fn run_fit(bin: &Path, fit_toml: &Path, threads: usize) -> std::process::Output {
    // The tiny 4-iteration fit does not converge `tau`; the source
    // convergence check is orthogonal to the CAS properties under test, so a
    // downstream fit (one that starts `from_mle`) lifts it. A fit with no
    // source has no verdict to lift, and `fit run` refuses the flag there.
    let starts_from_a_fit = std::fs::read_to_string(fit_toml)
        .expect("read fit toml")
        .contains("from_mle");
    let lift: &[&str] = if starts_from_a_fit { &["--allow-nonconverged-source"] } else { &[] };
    Command::new(bin).env_remove("CLICOLOR_FORCE")
        .arg("fit").arg("run").arg(fit_toml)
        .args(lift)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("RAYON_NUM_THREADS", threads.to_string())
        .output()
        .expect("camdl fit run must spawn")
}

/// Like `run_fit` but caps the pool via the `--parallel` FLAG (gh#162) rather
/// than the RAYON_NUM_THREADS env — exercises the flag's wiring end to end.
fn run_fit_parallel(bin: &Path, fit_toml: &Path, parallel: usize) -> std::process::Output {
    Command::new(bin).env_remove("CLICOLOR_FORCE")
        .arg("fit").arg("run").arg(fit_toml)
        .arg("--parallel").arg(parallel.to_string())
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("camdl fit run must spawn")
}

/// Every fit run stored under the store `out`: (run folder, run id).
fn method_leaves(out: &Path) -> Vec<(PathBuf, String)> {
    let mut found: Vec<(PathBuf, String)> = fit_store_support::runs(out)
        .into_iter()
        .map(|r| {
            let id = fit_store_support::run_id(&r);
            (r, id)
        })
        .collect();
    found.sort();
    found
}

/// The one run a single-file fit stored.
fn only_leaf(out: &Path) -> (PathBuf, String) {
    let leaves = method_leaves(out);
    assert_eq!(leaves.len(), 1, "expected one fit run, got {leaves:?}");
    leaves.into_iter().next().unwrap()
}

/// Read the upstream fit's θ̂ from its `mle_params.toml`, stripping the
/// `[provenance]` block (a wall-clock timestamp + a fit_hash of the
/// path-differing fit.toml, both of which legitimately differ between runs).
/// Used to assert parallel-invariance of the estimate.
fn read_scout_mle(out: &Path) -> String {
    let (leaf, _) = only_leaf(out);
    let full = std::fs::read_to_string(leaf.join("mle_params.toml")).unwrap();
    full.split("[provenance]").next().unwrap_or(&full).trim().to_string()
}

/// Property 1: the downstream file's run shares the upstream's fit problem;
/// editing its `[method]` adds only a run of its own; re-running the upstream
/// file is a cache hit that keeps its run id.
#[test]
fn chained_method_reuse_only_rekeys_the_downstream() {
    let bin = bin();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("out");
    let data = write_data(tmp.path());

    // Run 1: the upstream fit.
    let scout = write_scout_toml(tmp.path(), &out, &data);
    let r1 = run_fit(&bin, &scout, 1);
    assert!(r1.status.success(), "scout run failed: {}", String::from_utf8_lossy(&r1.stderr));
    let (scout_leaf, scout_id) = only_leaf(&out);

    // Run 2: the downstream fit, started at the upstream's point estimate.
    let post4 = write_posterior_toml(tmp.path(), &out, &data, &scout_leaf, 4);
    let r2 = run_fit(&bin, &post4, 1);
    assert!(r2.status.success(), "posterior run failed: {}", String::from_utf8_lossy(&r2.stderr));
    let leaves2 = method_leaves(&out);
    assert_eq!(leaves2.len(), 2, "one problem, two methods, two runs: {leaves2:?}");
    let post4_id = leaves2.iter().find(|(_, id)| *id != scout_id)
        .map(|(_, id)| id.clone()).expect("the downstream run");
    // The two runs share one stored problem: its id hashes the problem, not
    // `[method]`, and the file stem plays no part in it.
    let problems: std::collections::BTreeSet<PathBuf> =
        leaves2.iter().map(|(run, _)| fit_store_support::problem_of(run)).collect();
    assert_eq!(problems.len(), 1,
        "the two runs must sit in one fit problem; problems: {problems:?}");
    assert_eq!(fit_store_support::problems(&out).len(), 1,
        "the store must hold exactly one fit problem: {:?}", fit_store_support::problems(&out));
    let problem_ids: std::collections::BTreeSet<String> = leaves2.iter()
        .map(|(run, _)| fit_store_support::run_inputs(run)["problem_id"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(problem_ids.len(), 1, "both runs must name one problem id: {problem_ids:?}");

    // Run 3: the upstream again, unchanged → served from cache, same run_id.
    let r3 = run_fit(&bin, &scout, 1);
    assert!(r3.status.success(), "scout rerun failed: {}", String::from_utf8_lossy(&r3.stderr));
    let stderr3 = String::from_utf8_lossy(&r3.stderr);
    assert!(stderr3.to_lowercase().contains("cache hit"),
        "an unchanged file must be a cache hit; stderr:\n{stderr3}");

    // Run 4: edit ONLY the downstream's config (iterations 4 → 8).
    let post8 = write_posterior_toml(tmp.path(), &out, &data, &scout_leaf, 8);
    let r4 = run_fit(&bin, &post8, 1);
    assert!(r4.status.success(), "edited posterior run failed: {}", String::from_utf8_lossy(&r4.stderr));
    let leaves4 = method_leaves(&out);
    let ids: std::collections::BTreeSet<&str> = leaves4.iter().map(|(_, id)| id.as_str()).collect();

    // The upstream is untouched (still exactly its one run), the first
    // downstream run survives, and the edit added a distinct third.
    assert!(ids.contains(scout_id.as_str()),
        "editing the downstream must not re-key the upstream; scout={scout_id} ids={ids:?}");
    assert!(ids.contains(post4_id.as_str()),
        "the iter=4 downstream run must survive; post4={post4_id} ids={ids:?}");
    assert_eq!(ids.len(), 3,
        "editing the downstream (iters 4→8) must produce a distinct third run; got {ids:?}");
    // And every downstream run carries a dep on the upstream run's point
    // estimate, so a re-run upstream re-keys them (gh#541).
    for (leaf, id) in &leaves4 {
        if *id == scout_id { continue; }
        let rec = fit_store_support::json(&leaf.join("run.json"));
        let deps_upstream = rec["deps"].as_array().expect("run.json carries deps").iter().any(|d| {
            d["run_id"].as_str() == Some(scout_id.as_str())
                && d["file"].as_str().is_some_and(|f| f.starts_with("fit_state.toml"))
        });
        assert!(deps_upstream,
            "downstream run {} must dep on the upstream's fit_state.toml: {rec}", leaf.display());
    }
}

/// gh#901: a fit run's `run_inputs.json` names its algorithm once, under
/// `method`, and not also under `stage`, so a consumer has one key to read.
/// The run's folder sits in a `runs/<method>-<id>/` group named by the same
/// word, and its `identity.json` gives the kind `fit_run`.
#[test]
fn a_method_leaf_names_its_algorithm_once() {
    let bin = bin();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("out");
    let data = write_data(tmp.path());
    let scout = write_scout_toml(tmp.path(), &out, &data);
    let r = run_fit(&bin, &scout, 1);
    assert!(r.status.success(), "scout run failed: {}", String::from_utf8_lossy(&r.stderr));

    let (leaf, _) = only_leaf(&out);
    let inputs = fit_store_support::run_inputs(&leaf);
    let text = inputs.to_string();
    let inputs = inputs.as_object().expect("run_inputs.json is an object");
    assert_eq!(inputs.get("method").and_then(|v| v.as_str()), Some("if2"),
        "run_inputs.json must name the algorithm under `method`:\n{text}");
    assert!(inputs.get("stage").is_none(),
        "and must not also carry it under `stage` (gh#901):\n{text}");

    // The run's method group is named by the same word.
    assert_eq!(fit_store_support::runs_of_method(&out, "if2"), vec![leaf.clone()],
        "the run must sit in a `runs/if2-<id>/` group");

    let ident = fit_store_support::json(&leaf.join("identity.json"));
    assert_eq!(ident["kind"].as_str(), Some("fit_run"), "identity.json:\n{ident}");
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
    let toml1 = write_scout_toml(tmp.path(), &out1, &data);
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
        "θ̂ (mle_params.toml) must be bit-identical at --parallel 1 vs 8 \
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
    let toml1 = write_scout_toml(tmp.path(), &out1, &data);
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

/// Property 3: `fit run` announces the store its run lands in. The announced
/// `output:` path exists, is the store root the fit file names, and holds the
/// run's problem folder, which holds an `if2-<id>` method group.
#[test]
fn fit_run_announces_the_real_leaf_directory() {
    let bin = bin();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("out");
    let data = write_data(tmp.path());

    let toml = write_scout_toml(tmp.path(), &out, &data);
    let r = run_fit(&bin, &toml, 1);
    assert!(r.status.success(), "fit run failed: {}", String::from_utf8_lossy(&r.stderr));
    let stderr = String::from_utf8_lossy(&r.stderr);

    // Parse the announced `output:` directory from the startup block.
    let announced = stderr
        .lines()
        .find_map(|l| l.trim_start().strip_prefix("output:"))
        .map(|s| PathBuf::from(s.trim()))
        .unwrap_or_else(|| panic!("no `output:` line in fit run stderr:\n{stderr}"));

    // The announced dir must exist on disk.
    assert!(
        announced.is_dir(),
        "announced output dir must exist on disk: {}\nstderr:\n{stderr}",
        announced.display(),
    );

    assert_eq!(
        announced.canonicalize().unwrap(),
        out.canonicalize().unwrap(),
        "the announced `output:` dir ({}) must be the store the fit file names ({})",
        announced.display(),
        out.display(),
    );

    // The store holds the run, inside its problem, inside an `if2-<id>` group.
    let (run, _) = only_leaf(&announced);
    let problem = fit_store_support::problem_of(&run);
    assert!(problem.starts_with(&announced), "run {} is not under {}", run.display(), announced.display());
    let has_method_child = std::fs::read_dir(problem.join("runs"))
        .unwrap()
        .flatten()
        .any(|e| e.path().is_dir() && e.file_name().to_string_lossy().starts_with("if2-"));
    assert!(
        has_method_child,
        "problem {} must contain the `if2-<id>` method group",
        problem.display(),
    );
}
