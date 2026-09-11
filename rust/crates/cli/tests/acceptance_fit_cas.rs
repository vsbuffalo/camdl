//! gh#147 (M3.2) — content-addressed fit-stage acceptance gates.
//!
//! Two properties the fit-stage CAS migration must guarantee, exercised
//! end-to-end against the release `camdl` binary:
//!
//! 1. **Chained-method reuse** — a second file that warm-starts from the
//!    first (`starts = { from_mle = <leaf> }`) shares the first's fit-level
//!    hash, because that level hashes the problem alone (the two segments
//!    are siblings named by their stems, `scout-<h8>` and `posterior-<h8>`);
//!    editing the second file's `[method]` re-keys only its own leaf, and
//!    re-running the first file is a cache hit that keeps its `run_id`. This
//!    is the `FitDigest`-excludes-`[method]` factoring working: the expensive
//!    upstream fit is cached across downstream-config iteration.
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

/// The problem half both files share.
fn problem_half(out: &Path, data: &Path) -> String {
    format!(
        r#"output_dir = "{out}"

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
    Command::new(bin)
        .arg("fit").arg("run").arg(fit_toml)
        // The tiny 4-iteration fit does not converge `tau`; the source
        // convergence check is orthogonal to the CAS properties under test,
        // so lift it and let the downstream fit start from it.
        .arg("--allow-nonconverged-source")
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
        .arg("--parallel").arg(parallel.to_string())
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("camdl fit run must spawn")
}

/// Every CAS fit-stage leaf under `out/fits/`: (leaf dir, run_id), read from
/// each `run.json` (kind = `fit_stage`).
fn stage_leaves(out: &Path) -> Vec<(PathBuf, String)> {
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
                let leaf = p.parent().map(Path::to_path_buf).unwrap_or_default();
                found.push((leaf, run_id));
            }
        }
    }
    found.sort();
    found
}

/// The one leaf a single-file run wrote.
fn only_leaf(out: &Path) -> (PathBuf, String) {
    let leaves = stage_leaves(out);
    assert_eq!(leaves.len(), 1, "expected one method leaf, got {leaves:?}");
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

/// Property 1: the downstream file shares the upstream's fit-level hash;
/// editing its `[method]` re-keys only its own leaf; re-running the upstream
/// file is a cache hit that keeps its `run_id`.
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
    let leaves2 = stage_leaves(&out);
    assert_eq!(leaves2.len(), 2, "one problem, two methods, two leaves: {leaves2:?}");
    let post4_id = leaves2.iter().find(|(_, id)| *id != scout_id)
        .map(|(_, id)| id.clone()).expect("the downstream leaf");
    // The two segments share the fit-level hash (the `-<h8>` suffix): that
    // level hashes the problem, and the file stem only labels the directory.
    let segment_h8 = |leaf: &Path| -> String {
        let seg = leaf.parent().and_then(|p| p.parent()).unwrap();
        let name = seg.file_name().unwrap().to_string_lossy().into_owned();
        name.rsplit('-').next().unwrap().to_string()
    };
    let h8s: std::collections::BTreeSet<String> =
        leaves2.iter().map(|(leaf, _)| segment_h8(leaf)).collect();
    assert_eq!(h8s.len(), 1,
        "the two method leaves must share the fit-level hash; segments: {:?}",
        leaves2.iter().map(|(l, _)| l.parent().and_then(|p| p.parent()).unwrap().to_path_buf()).collect::<Vec<_>>());

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
    let leaves4 = stage_leaves(&out);
    let ids: std::collections::BTreeSet<&str> = leaves4.iter().map(|(_, id)| id.as_str()).collect();

    // The upstream is untouched (still exactly its one leaf), the first
    // downstream leaf survives, and the edit added a distinct third.
    assert!(ids.contains(scout_id.as_str()),
        "editing the downstream must not re-key the upstream; scout={scout_id} ids={ids:?}");
    assert!(ids.contains(post4_id.as_str()),
        "the iter=4 downstream leaf must survive; post4={post4_id} ids={ids:?}");
    assert_eq!(ids.len(), 3,
        "editing the downstream (iters 4→8) must produce a distinct third leaf; got {ids:?}");
    // And every downstream leaf carries a dep on the upstream's point estimate,
    // so a re-run upstream re-keys them (gh#541).
    for (leaf, id) in &leaves4 {
        if *id == scout_id { continue; }
        let rec = std::fs::read_to_string(leaf.join("run.json")).unwrap();
        assert!(rec.contains("fit_state.toml"),
            "downstream leaf {} must dep on the upstream's fit_state.toml: {rec}", leaf.display());
    }
}

/// gh#901: a method leaf's `run.json` names its algorithm once, under
/// `method`.
///
/// It used to carry the same string twice — `inputs.stage` beside
/// `inputs.method`, both set from `Algorithm::method_name()` — so the leaf
/// published one fact under two vocabularies and a consumer could key on
/// either. Two-sided: `method` is the algorithm the fit declared, and `stage`
/// is gone from `inputs` rather than kept beside it.
///
/// The `kind` string stays `fit_stage`: it is read back from every store on
/// disk, and renaming it would orphan them for a word that no consumer
/// interprets as a workflow stage.
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
    let text = std::fs::read_to_string(leaf.join("run.json")).unwrap();
    let rec: serde_json::Value = serde_json::from_str(&text).unwrap();
    let inputs = rec["inputs"].as_object().expect("run.json carries an inputs object");
    assert_eq!(inputs.get("method").and_then(|v| v.as_str()), Some("if2"),
        "inputs must name the algorithm under `method`:\n{text}");
    assert!(inputs.get("stage").is_none(),
        "and must not also carry it under `stage` (gh#901):\n{text}");

    // The middle store level is `method` too — same word, same leaf.
    let levels = rec["levels"].as_array().expect("run.json carries levels");
    let names: Vec<&str> = levels.iter().filter_map(|l| l["name"].as_str()).collect();
    assert_eq!(names, vec!["fit", "method", "seed"], "level names:\n{text}");

    // Deliberately kept: the artifact-kind string.
    assert_eq!(rec["kind"].as_str(), Some("fit_stage"),
        "the kind string is read back from every existing store and is not \
         renamed by gh#901:\n{text}");
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

/// Property 3 (Q2): `fit run` must announce the directory its method leaves
/// actually land in. The leaves are written by `resolve_fit_stage` under the
/// `FitDigest` fit-level segment (`fits/{stem}-{h8}/`); the announced
/// `output:` line must be exactly that segment — i.e. the announced path
/// EXISTS and is the parent of the `{method}-{h8}` leaf dirs.
///
/// The announced path and the leaves now share one basis: the `FitDigest`
/// fit-level hash via `fit_segment_dir` (real and synthetic alike), so the
/// announced path always exists and parents the method leaves.
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

    // The announced dir must exist on disk (it is the fit segment the leaves
    // land under, not a divergent directory).
    assert!(
        announced.is_dir(),
        "announced output dir must exist on disk: {}\nstderr:\n{stderr}",
        announced.display(),
    );

    // Locate the actual method leaf (the dir holding a fit_stage run.json) and
    // confirm the announced dir is its grandparent — the fit segment that
    // `store_path` factors as fits/{fit}/{method}/{seed}/.
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

    // Leaf factoring: fits/{fit}/{method}/{seed}/run.json — so the fit
    // segment is the seed dir's grandparent (= leaf_dir.parent().parent()).
    let fit_segment = leaf_dir
        .parent()
        .and_then(|p| p.parent())
        .unwrap_or_else(|| panic!("leaf {} has no fit-segment grandparent", leaf_dir.display()));

    assert_eq!(
        announced.canonicalize().unwrap(),
        fit_segment.canonicalize().unwrap(),
        "the announced `output:` dir ({}) must equal the actual fit segment \
         that holds the method leaf dirs ({})",
        announced.display(),
        fit_segment.display(),
    );

    // And the segment really contains the method tree (an `if2-<h8>` child,
    // labelled by the algorithm).
    let has_method_child = std::fs::read_dir(fit_segment)
        .unwrap()
        .flatten()
        .any(|e| e.path().is_dir() && e.file_name().to_string_lossy().starts_with("if2-"));
    assert!(
        has_method_child,
        "announced fit segment {} must contain the `if2-<h8>` method leaf dir",
        fit_segment.display(),
    );
}
