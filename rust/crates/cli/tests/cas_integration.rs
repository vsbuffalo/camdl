//! End-to-end tests for `camdl simulate --cas` and `camdl list/show/cat`.
//!
//! These shell out to the built `camdl` binary in `target/release/`
//! and exercise the full pipeline: real hash computation, real cache
//! lookups, real directory writes.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Path to the release-built binary. Tests assume a prior
/// `cargo build --release -p cli`, which happens automatically before
/// `cargo test` in CI. Skips the test if the binary is absent.
fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let bin = Path::new(&manifest)
        .join("../../target/release/camdl");
    bin
}

/// A golden IR with a baseline scenario that sets beta/gamma/N0/I0 —
/// suitable for `--cas --scenario baseline --seed N`.
fn golden_sir_basic() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../ocaml/golden/sir_basic.ir.json")
}

fn skip_if_missing_binary() -> Option<PathBuf> {
    let bin = binary();
    if !bin.exists() {
        eprintln!("skipping: camdl binary not built at {}", bin.display());
        return None;
    }
    Some(bin)
}

// ── New-format (runid::RunRecord) helpers ──────────────────────────────────

/// Read a sim leaf's `run.json` (a `RunRecord`) as JSON.
fn read_meta(dir: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(dir.join("run.json")).unwrap()).unwrap()
}

/// A named factored level's label (`model`/`config`/`params`/`scenario`/`seed`).
fn level_label<'a>(meta: &'a serde_json::Value, name: &str) -> &'a str {
    meta["levels"].as_array().unwrap().iter()
        .find(|l| l["name"] == name).unwrap_or_else(|| panic!("level {name} present"))
        ["label"].as_str().unwrap()
}

/// A named factored level's content hash (64 hex).
fn level_hash<'a>(meta: &'a serde_json::Value, name: &str) -> &'a str {
    meta["levels"].as_array().unwrap().iter()
        .find(|l| l["name"] == name).unwrap_or_else(|| panic!("level {name} present"))
        ["hash"].as_str().unwrap()
}

#[test]
fn cas_first_run_writes_cache_and_metadata() {
    let Some(bin) = skip_if_missing_binary() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");

    let status = Command::new(&bin)
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline",
               "--seed", "42",
               "--cas",
               "--output-dir", &output.to_string_lossy(),
               "-o", &tmp.path().join("traj.tsv").to_string_lossy()])
        .status()
        .expect("spawn");
    assert!(status.success(), "first --cas run should succeed");

    // Exactly one CAS entry under runs/
    let runs = output.join("sims");
    assert!(runs.exists(), "sims/ directory should exist");
    let seed_dirs: Vec<_> = walkdir(&runs).into_iter()
        .filter(|p| p.join("run.json").exists())
        .collect();
    assert_eq!(seed_dirs.len(), 1, "should have exactly one run dir");

    let dir = &seed_dirs[0];
    assert!(dir.join("traj.tsv").exists(), "traj.tsv should be written");
    assert!(dir.join("run.json").exists(), "run.json should be written");

    // run.json is a RunRecord: factored levels + run_id + provenance.
    let meta = read_meta(dir);
    assert_eq!(meta["kind"], "sim", "new-format RunRecord: kind=sim");
    assert_eq!(meta["run_id"].as_str().unwrap().len(), 64);
    assert_eq!(meta["status"], "completed");
    assert_eq!(level_label(&meta, "seed"), "seed_42");
    assert_eq!(level_label(&meta, "scenario"), "baseline");
    assert!(meta["provenance"]["argv"].as_array().unwrap().len() >= 4);
    assert!(meta["engine_version"].as_str().unwrap().contains('+'),
        "engine_version should include git hash suffix");
    assert!(meta["provenance"]["created_at"].as_str().is_some());
}

#[test]
fn cas_second_identical_run_is_cache_hit() {
    let Some(bin) = skip_if_missing_binary() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");

    let run_once = || {
        Command::new(&bin)
            .args(["simulate", &golden_sir_basic().to_string_lossy(),
                   "--scenario", "baseline",
                   "--seed", "42",
                   "--cas",
                   "--output-dir", &output.to_string_lossy(),
                   "-o", &tmp.path().join("traj.tsv").to_string_lossy()])
            .output()
            .expect("spawn")
    };

    let first = run_once();
    assert!(first.status.success());
    let stderr1 = String::from_utf8_lossy(&first.stderr);
    assert!(stderr1.contains("cached:"), "first run stderr should say 'cached:': {}", stderr1);

    // Wait long enough that the filesystem mtime would differ if rewritten
    let cache_path = walkdir(&output.join("sims")).into_iter()
        .find(|p| p.join("traj.tsv").exists()).unwrap()
        .join("traj.tsv");
    let mtime1 = std::fs::metadata(&cache_path).unwrap().modified().unwrap();

    std::thread::sleep(std::time::Duration::from_millis(10));

    let second = run_once();
    assert!(second.status.success());
    let stderr2 = String::from_utf8_lossy(&second.stderr);
    assert!(stderr2.contains("cache hit:"),
        "second run stderr should say 'cache hit:': {}", stderr2);

    // mtime unchanged — second run must not have re-written the file
    let mtime2 = std::fs::metadata(&cache_path).unwrap().modified().unwrap();
    assert_eq!(mtime1, mtime2, "cache hit must not overwrite traj.tsv");
}

#[test]
fn cas_different_seed_new_cache_entry() {
    let Some(bin) = skip_if_missing_binary() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");

    for seed in ["42", "43"] {
        let st = Command::new(&bin)
            .args(["simulate", &golden_sir_basic().to_string_lossy(),
                   "--scenario", "baseline",
                   "--seed", seed,
                   "--cas",
                   "--output-dir", &output.to_string_lossy(),
                   "-o", &tmp.path().join("traj.tsv").to_string_lossy()])
            .status().expect("spawn");
        assert!(st.success());
    }

    let dirs: Vec<_> = walkdir(&output.join("sims")).into_iter()
        .filter(|p| p.join("run.json").exists()).collect();
    assert_eq!(dirs.len(), 2, "should have two separate seed dirs");

    // Seed segments are `seed_{base}-{seed_h8}` (no level is label-only).
    let seeds: Vec<String> = dirs.iter()
        .map(|d| d.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert!(seeds.iter().any(|n| n.starts_with("seed_42-")), "got {seeds:?}");
    assert!(seeds.iter().any(|n| n.starts_with("seed_43-")), "got {seeds:?}");
}

/// gh#135 regression. Two structurally different models under identical
/// params/backend/dt/seed must NOT collide to one CAS entry. Pre-fix,
/// `model_hash` scanned the envelope's top level (keys live under
/// `model`), hashed nothing, and returned SHA256("") for every model, so
/// `sim_hash` was blind to structure: the second model was silently
/// served the first model's cached trajectory.
///
/// The two models share a basename (`model.ir.json`) so the path's
/// model-stem prefix is identical — the ONLY thing that can separate
/// them is `model_hash`. They differ only in the `recovery` transition's
/// rate (×2), a structural field hashed by `model_hash` but NOT a
/// parameter value, so `base_params_canonical` is identical between them.
#[test]
fn cas_different_models_do_not_collide() {
    let Some(bin) = skip_if_missing_binary() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");

    // v1 = golden as-is; v2 = recovery rate doubled (structural change).
    let v1: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(golden_sir_basic()).unwrap()).unwrap();
    let mut v2 = v1.clone();
    {
        let transitions = v2["model"]["transitions"].as_array_mut()
            .expect("model.transitions array");
        let recovery = transitions.iter_mut()
            .find(|t| t["name"] == serde_json::json!("recovery"))
            .expect("a `recovery` transition");
        let old_rate = recovery["rate"].clone();
        recovery["rate"] = serde_json::json!({
            "bin_op": { "op": "mul", "left": old_rate, "right": { "const": 2.0 } }
        });
    }
    assert_ne!(v1["model"]["transitions"], v2["model"]["transitions"],
        "test setup: the two models must actually differ structurally");

    // Identical basename in separate dirs → identical model-stem prefix.
    let dir_a = tmp.path().join("a");
    let dir_b = tmp.path().join("b");
    std::fs::create_dir_all(&dir_a).unwrap();
    std::fs::create_dir_all(&dir_b).unwrap();
    let path_a = dir_a.join("model.ir.json");
    let path_b = dir_b.join("model.ir.json");
    std::fs::write(&path_a, serde_json::to_string(&v1).unwrap()).unwrap();
    std::fs::write(&path_b, serde_json::to_string(&v2).unwrap()).unwrap();

    let run = |model: &Path| {
        let st = Command::new(&bin)
            .args(["simulate", &model.to_string_lossy(),
                   "--scenario", "baseline",
                   "--seed", "1",
                   "--cas",
                   "--output-dir", &output.to_string_lossy(),
                   "-o", &tmp.path().join("traj.tsv").to_string_lossy()])
            .status().expect("spawn");
        assert!(st.success(), "simulate --cas should succeed");
    };
    run(&path_a);
    run(&path_b);

    // Two distinct cache entries — not one collision.
    let dirs: Vec<_> = walkdir(&output.join("sims")).into_iter()
        .filter(|p| p.join("run.json").exists()).collect();
    assert_eq!(dirs.len(), 2,
        "two structurally different models must produce two CAS entries, \
         not collide to one (gh#135)");

    // The model LEVEL hash (structural whole-IR digest) must be recorded,
    // 64-hex, and distinct between the two structurally different models.
    let mut model_hashes: Vec<String> = dirs.iter().map(|d| {
        let meta = read_meta(d);
        level_hash(&meta, "model").to_string()
    }).collect();
    model_hashes.sort();
    for mh in &model_hashes {
        assert_eq!(mh.len(), 64, "model level hash is 64-hex");
    }
    assert_ne!(model_hashes[0], model_hashes[1],
        "two structurally different models must record distinct model level \
         hashes (gh#135/gh#147)");

    // And the trajectories themselves must differ — the symptom users saw.
    let trajs: Vec<Vec<u8>> = dirs.iter()
        .map(|d| std::fs::read(d.join("traj.tsv")).unwrap()).collect();
    assert_ne!(trajs[0], trajs[1],
        "doubling the recovery rate must change the trajectory; identical \
         bytes means one model was served the other's cached result (gh#135)");
}

#[test]
fn cas_rejects_multi_seeds() {
    let Some(bin) = skip_if_missing_binary() else { return; };
    let tmp = tempfile::tempdir().unwrap();

    let out = Command::new(&bin)
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline",
               "--seeds", "1:3",
               "--cas",
               "--output-dir", &tmp.path().to_string_lossy()])
        .output()
        .expect("spawn");
    assert!(!out.status.success(), "multi-seed + --cas should fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("--cas supports single runs only"),
        "error should name the limitation: {}", stderr);
    assert!(stderr.contains("batch run"),
        "error should hint at `batch run`: {}", stderr);
}

#[test]
fn list_shows_cached_runs() {
    let Some(bin) = skip_if_missing_binary() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");

    // Cache two runs
    for seed in ["42", "99"] {
        Command::new(&bin)
            .args(["simulate", &golden_sir_basic().to_string_lossy(),
                   "--scenario", "baseline",
                   "--seed", seed,
                   "--cas",
                   "--output-dir", &output.to_string_lossy(),
                   "-o", &tmp.path().join("traj.tsv").to_string_lossy()])
            .status().expect("spawn");
    }

    // `camdl list` should find both
    let out = Command::new(&bin)
        .args(["list", &output.to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success(), "list should succeed");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("seed_42"), "list should include seed_42: {}", stdout);
    assert!(stdout.contains("seed_99"), "list should include seed_99: {}", stdout);
    assert!(stdout.contains("baseline"), "list should show scenario name");
}

/// `--starts-from` was removed in the 2026-05-25 CLI UX rev 2 M-1
/// break. Users get an actionable error pointing at the replacement
/// (`--init from_mle --mle <fit-dir>`), and `--init from_mle --mle
/// <hash>` resolves a short-hash prefix to the matching fit-stage
/// directory (preserving Hardening #9's resolver behaviour under the
/// new spelling).
#[test]
fn starts_from_resolves_short_hash() {
    let Some(bin) = skip_if_missing_binary() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let results = tmp.path().join("results");
    // Fake a stage dir with a known-hash run.json. The resolver
    // walks results/fits/**, so we need to place the run.json
    // under that structure.
    let stage = results.join("fits").join("demo-abc12345")
        .join("real").join("fit_1").join("scout");
    std::fs::create_dir_all(&stage).unwrap();
    let target_hash = "deadbeefc0ffee00000000000000000000000000000000000000000000000000";
    let run_json = format!(r#"{{
        "hash": "{}",
        "version": "0.1.0+test","created_at": "2026-04-19T12:00:00Z",
        "argv": [],"status": {{"completed": {{"wall_time_seconds": 1.0}}}},
        "kind": {{"kind":"fit-stage","fit_hash":"f","stage":"scout",
                 "method":"if2","backend":"chain_binomial",
                 "seed":1,"n_chains":2}}
    }}"#, target_hash);
    std::fs::write(stage.join("run.json"), run_json).unwrap();

    // Run `camdl fit where --help` just to prove --starts-from
    // resolution runs on the binary path; we don't need a real fit.
    // Instead, exercise the resolver directly via the browse library
    // behavior by using an ad-hoc subcommand path — but since we
    // don't expose resolve_stage_by_hash as a CLI command, do a
    // smoke test of the `--starts-from <hash>` entry by running
    // `fit where` which doesn't consume --starts-from. So the
    // meaningful test here is: the binary accepts `--starts-from
    // deadbeef` on a fit.toml and doesn't reject on the path check.
    // We can confirm via `fit run` dry-run... but we have no dry-run.
    //
    // Compromise: test the hash resolver by calling `fit where` is a
    // no-op for --starts-from. Instead, exercise the feature end-to-
    // end by running cmd_fit_run_v2 with a minimal fit.toml that
    // references --starts-from <hash>. If resolution fails, the run
    // exits 1 before it finishes doing useful work; if resolution
    // succeeds it proceeds to an actual fit (which we don't want).
    //
    // Keep it simple: just verify that passing --starts-from <bad
    // prefix> errors with our custom message, and --starts-from <good
    // prefix> runs at least to the point of config validation.
    // Changing from working directory to `results` parent so the
    // default `./results` resolver finds our stage.
    std::env::set_current_dir(tmp.path()).unwrap();

    // Bad hash: should error with our message.
    let ir = tmp.path().join("dummy.ir.json");
    std::fs::write(&ir, r#"{"compartments":[],"parameters":[]}"#).unwrap();
    let data = tmp.path().join("cases.tsv");
    std::fs::write(&data, "time\tcases\n1\t5\n").unwrap();
    let fit_toml = tmp.path().join("f.toml");
    std::fs::write(&fit_toml, format!(r#"
output_dir = "results"
[model]
camdl = "{}"
[data.observations]
cases = "{}"
[estimate]
beta = {{ bounds = [0.01, 2.0] }}
[fixed]
N0 = 1000
[stages.refine]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 50
iterations = 3
cooling = 0.7
init_mle = "{{use CLI}}"
"#, ir.display(), data.display())).unwrap();

    // Removed flag: `--starts-from` produces the actionable error
    // from the M-1 break (proposal §"Migration"), regardless of value.
    let out = Command::new(&bin)
        .current_dir(tmp.path())
        .args(["fit", "run", &fit_toml.to_string_lossy(),
               "--stage", "refine",
               "--starts-from", "deadbeef"])
        .output().expect("spawn");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(),
        "--starts-from must fail with the removed-flag error");
    assert!(stderr.contains("--starts-from is no longer accepted"),
        "expected actionable removed-flag error, got: {}", stderr);
    assert!(stderr.contains("--init from_mle --mle"),
        "removed-flag error must spell out the replacement, got: {}",
        stderr);

    // Bad hash with the new spelling: same short-hash resolver, same
    // actionable error.
    let out = Command::new(&bin)
        .current_dir(tmp.path())
        .args(["fit", "run", &fit_toml.to_string_lossy(),
               "--stage", "refine",
               "--init", "from_mle",
               "--mle", "zzzznonexistent"])
        .output().expect("spawn");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "bad hash must fail");
    assert!(stderr.contains("no fit stage matching hash prefix"),
        "expected 'no fit stage matching hash prefix', got: {}", stderr);

    // Good hash: resolves to the fake stage we planted via the new
    // `--init from_mle --mle <hash>` spelling. Resolution happens
    // before the fit actually does anything expensive, so verifying
    // the success path means checking that we get past arg parsing —
    // the fit itself may still fail downstream (the model IR is
    // empty), but the warm-start lookup succeeded.
    let out = Command::new(&bin)
        .current_dir(tmp.path())
        .args(["fit", "run", &fit_toml.to_string_lossy(),
               "--stage", "refine",
               "--init", "from_mle",
               "--mle", "deadbeef"])
        .output().expect("spawn");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("no fit stage matching hash prefix"),
        "short-hash 'deadbeef' should resolve to the planted stage, \
         got: {}", stderr);
}

/// `camdl fit where fit.toml` should print the fit root (post-
/// hardening #8). `camdl fit where fit.toml --seed N` should extend
/// that path with `real/fit_N`.
#[test]
fn fit_where_prints_fit_root_and_cell_dir() {
    let Some(bin) = skip_if_missing_binary() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    // gh#35: fit where now runs the same validation depth as fit run,
    // so the IR must declare every parameter the fit.toml estimates or
    // fixes. Pre-gh#35 this fixture was a hand-written minimal IR with
    // an empty parameters list, which only worked because `where`
    // didn't validate — exactly the bug gh#35 closed. Use the real
    // sir_basic golden (4 params: beta, gamma, N0, I0) and align the
    // fit.toml below.
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let golden_ir = format!("{}/../../../ir/golden/sir_basic.ir.json", manifest);
    let ir = tmp.path().join("dummy.ir.json");
    std::fs::copy(&golden_ir, &ir).expect("copy golden IR");
    let data = tmp.path().join("cases.tsv");
    std::fs::write(&data, "time\tcases\n1\t5\n").unwrap();
    let fit_toml = tmp.path().join("myfit.toml");
    std::fs::write(&fit_toml, format!(r#"
output_dir = "{}/out"

[model]
camdl = "{}"

[data.observations]
cases = "{}"

[estimate]
beta = {{ bounds = [0.01, 2.0], start = 0.3 }}

[fixed]
gamma = 0.1
N0 = 1000
I0 = 10

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 50
iterations = 3
cooling = 0.7
"#, tmp.path().display(), ir.display(), data.display())).unwrap();

    // No --seed: should print the fit root, ending with `/out/fits/myfit-<8hash>`
    let out = Command::new(&bin)
        .args(["fit", "where", &fit_toml.to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success(), "stderr: {}",
        String::from_utf8_lossy(&out.stderr));
    let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
    assert!(path.contains("/out/fits/myfit-"),
        "expected .../out/fits/myfit-<hash>, got {}", path);

    // --seed 42: should append real/fit_42.
    let out = Command::new(&bin)
        .args(["fit", "where", &fit_toml.to_string_lossy(), "--seed", "42"])
        .output().expect("spawn");
    assert!(out.status.success());
    let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
    assert!(path.ends_with("/real/fit_42"),
        "expected .../real/fit_42, got {}", path);
}

/// `camdl list --kind fit` should hide sim rows entirely; `--kind sim`
/// should hide fit rows. Covers docs/dev/notes/2026-04-20-unified-output-tree-cleanup.md:L2.
#[test]
fn list_kind_filter_isolates_sections() {
    let Some(bin) = skip_if_missing_binary() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");

    // Cache one sim.
    Command::new(&bin)
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline", "--seed", "7", "--cas",
               "--output-dir", &output.to_string_lossy(),
               "-o", &tmp.path().join("t.tsv").to_string_lossy()])
        .status().expect("spawn");
    // Synthesise a fit.
    let fit_dir = output.join("fits").join("demo-abc12345");
    std::fs::create_dir_all(&fit_dir).unwrap();
    let run_json = r#"{
        "hash": "abc12345deadbeef0000000000000000000000000000000000000000abc12345",
        "version":"0.1.0+test","created_at":"2026-04-19T12:00:00Z",
        "argv":["camdl","fit","run","demo.toml"],"status":{"completed":{"wall_time_seconds":1.0}},
        "kind": {"kind":"fit","model":"demo.camdl","model_hash":"m",
        "fit_toml_path":"demo.toml","fit_toml_hash":"h",
        "data_hashes":{},"estimated":["beta"],"fixed":{},"stages_declared":["mle"]}
    }"#;
    std::fs::write(fit_dir.join("run.json"), run_json).unwrap();

    let fit_only = Command::new(&bin)
        .args(["list", "--kind", "fit", &output.to_string_lossy()])
        .output().expect("spawn");
    let s = String::from_utf8_lossy(&fit_only.stdout);
    assert!(s.contains("demo"), "--kind fit should show fits: {}", s);
    assert!(!s.contains("seed_7"), "--kind fit must hide sims: {}", s);

    let sim_only = Command::new(&bin)
        .args(["list", "--kind", "sim", &output.to_string_lossy()])
        .output().expect("spawn");
    let s = String::from_utf8_lossy(&sim_only.stdout);
    assert!(s.contains("seed_7"), "--kind sim should show sims: {}", s);
    assert!(!s.contains("demo-abc12345"), "--kind sim must hide fits: {}", s);
}

/// Regression guard for the unified output tree: `camdl list` must
/// render a fits section when `output/fits/<...>/run.json` exists,
/// independent of whether any sim runs are cached. We synthesize a
/// handcrafted `run.json` rather than running a full `camdl fit run`
/// — that keeps the test fast and orthogonal to fit-runner behaviour.
#[test]
fn list_shows_fit_entries() {
    let Some(bin) = skip_if_missing_binary() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");
    let fit_dir = output.join("fits").join("demo-abc12345");
    std::fs::create_dir_all(&fit_dir).unwrap();

    // Minimal valid Run + RunKind::Fit JSON matching run_meta.rs schema.
    let run_json = r#"{
        "hash": "abc12345deadbeef0000000000000000000000000000000000000000abc12345",
        "version": "0.1.0+test",
        "created_at": "2026-04-19T12:00:00Z",
        "argv": ["camdl","fit","run","demo.toml"],
        "status": {"completed": {"wall_time_seconds": 3.2}},
        "kind": {
            "kind": "fit",
            "model": "demo.camdl",
            "model_hash": "m000",
            "fit_toml_path": "demo.toml",
            "fit_toml_hash": "h000",
            "data_hashes": {"cases": "d000"},
            "estimated": ["beta","gamma"],
            "fixed": {"N0": 1000.0},
            "stages_declared": ["scout","refine"]
        }
    }"#;
    std::fs::write(fit_dir.join("run.json"), run_json).unwrap();

    let out = Command::new(&bin)
        .args(["list", &output.to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success(), "list should succeed: stderr={:?}",
        String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let all = format!("{}{}", stdout, stderr);
    assert!(all.contains("fits"), "list output must include a 'fits' section: {}", all);
    assert!(stdout.contains("demo"),
        "fit stem should appear in table: {}", stdout);
    assert!(stdout.contains("scout,refine"),
        "fit STAGES column should show declared stages: {}", stdout);
}

/// Tamper-with-metadata regression: if `run.json`'s stored hash no
/// longer matches the current sim config, `camdl simulate --cas`
/// should print a "stale cache" warning and re-run, not silently
/// serve the old trajectory.
#[test]
fn cas_stale_metadata_warns_and_reruns() {
    let Some(bin) = skip_if_missing_binary() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");

    let run_once = || {
        Command::new(&bin)
            .args(["simulate", &golden_sir_basic().to_string_lossy(),
                   "--scenario", "baseline",
                   "--seed", "42",
                   "--cas",
                   "--output-dir", &output.to_string_lossy(),
                   "-o", &tmp.path().join("traj.tsv").to_string_lossy()])
            .output().expect("spawn")
    };

    let _ = run_once();
    // Locate the run dir, then tamper with traj.tsv so the exact-set manifest
    // (size) no longer matches — the tiered-integrity lookup returns Stale and
    // the run is recomputed (never served corrupt bytes).
    let dir = walkdir(&output.join("sims")).into_iter()
        .find(|p| p.join("run.json").exists()).expect("one run");
    std::fs::write(dir.join("traj.tsv"), b"tampered-and-shorter").unwrap();

    // Second run should detect the stale cache and warn, then re-run.
    let out = run_once();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("stale cache"),
        "expected stale cache warning on stderr, got: {}", stderr);
    assert!(out.status.success(), "re-run should still succeed");
}

#[test]
fn show_resolves_fit_by_hash_prefix() {
    let Some(bin) = skip_if_missing_binary() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");
    let fit_dir = output.join("fits").join("demo-deadbeef");
    std::fs::create_dir_all(&fit_dir).unwrap();
    // hash starts with deadbeef so the short-hash lookup should match.
    let run_json = r#"{
        "hash":"deadbeefc0ffee00000000000000000000000000000000000000000000000000",
        "version":"0.1.0+test","created_at":"2026-04-19T12:00:00Z",
        "argv":["camdl","fit","run","demo.toml"],"status":{"completed":{"wall_time_seconds":1.0}},
        "kind":{"kind":"fit","model":"demo.camdl","model_hash":"m",
        "fit_toml_path":"demo.toml","fit_toml_hash":"h",
        "data_hashes":{},"estimated":["beta"],"fixed":{},"stages_declared":["mle"]}
    }"#;
    std::fs::write(fit_dir.join("run.json"), run_json).unwrap();
    let out = Command::new(&bin)
        .args(["show", "deadbee", "--root", &output.to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success(), "show by short-hash should resolve: stderr={}",
        String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("kind"));
    assert!(stdout.contains("fit"));
    assert!(stdout.contains("demo.camdl"));
}

#[test]
fn show_renders_fit_metadata() {
    let Some(bin) = skip_if_missing_binary() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");
    let fit_dir = output.join("fits").join("demo-abc12345");
    std::fs::create_dir_all(&fit_dir).unwrap();
    let run_json = r#"{
        "hash": "abc12345deadbeef0000000000000000000000000000000000000000abc12345",
        "version": "0.1.0+test",
        "created_at": "2026-04-19T12:00:00Z",
        "argv": ["camdl","fit","run","demo.toml"],
        "status": {"completed": {"wall_time_seconds": 3.2}},
        "kind": {
            "kind": "fit",
            "model": "demo.camdl",
            "model_hash": "m000",
            "fit_toml_path": "demo.toml",
            "fit_toml_hash": "h000",
            "data_hashes": {"cases": "d000"},
            "estimated": ["beta","gamma"],
            "fixed": {"N0": 1000.0},
            "stages_declared": ["scout","refine"]
        }
    }"#;
    std::fs::write(fit_dir.join("run.json"), run_json).unwrap();

    let out = Command::new(&bin)
        .args(["show", &fit_dir.to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success(), "show should succeed on a fit dir: stderr={}",
        String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("kind"),      "output should label the kind field");
    assert!(stdout.contains("fit"),       "output should say 'fit': {}", stdout);
    assert!(stdout.contains("demo.camdl"),"output should include model: {}", stdout);
    assert!(stdout.contains("scout, refine"), "output should list stages");
    assert!(stdout.contains("3.2"),       "output should show wall time");
}

#[test]
fn cat_emits_cached_trajectory() {
    let Some(bin) = skip_if_missing_binary() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");

    Command::new(&bin)
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline",
               "--seed", "42",
               "--cas",
               "--output-dir", &output.to_string_lossy(),
               "-o", &tmp.path().join("traj.tsv").to_string_lossy()])
        .status().expect("spawn");

    // Find the cached dir, derive the run_id short prefix.
    let dir = walkdir(&output.join("sims")).into_iter()
        .find(|p| p.join("run.json").exists()).unwrap();
    let meta = read_meta(&dir);
    let run_id = meta["run_id"].as_str().unwrap();
    let short = &run_id[..8];

    // `camdl cat <short>` uniquely resolves and emits the TSV
    let out = Command::new(&bin)
        .args(["cat", short, "--root", &output.to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success(), "cat short-hash should resolve uniquely");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.starts_with("# camdl"), "cat should emit trajectory header");
    assert!(stdout.contains("\tS\t") || stdout.contains("\tI\t"),
        "cat should include compartment columns");

    // Cached trajectory bytes should match the stdout of `cat`
    let cached = std::fs::read(dir.join("traj.tsv")).unwrap();
    assert_eq!(out.stdout, cached, "cat output must match cached bytes byte-for-byte");
}

#[test]
fn batch_sweep_records_sweep_point_in_run_json_and_manifest() {
    // Regression: before this fix, batch sweeps wrote run.json and
    // manifest entries with no record of the sweep parameter values —
    // you could see there were 8 distinct scen_hashes but not which
    // beta value produced which trajectory. This test runs a minimal
    // --batch with a 3-point sweep and asserts sweep_point is present.
    let Some(bin) = skip_if_missing_binary() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");

    // Use the golden sir_basic IR directly (no .camdl → camdlc step).
    // Write a params.toml with the non-swept params pinned.
    let params_path = tmp.path().join("params.toml");
    std::fs::write(&params_path, "beta = 0.3\ngamma = 0.1\nN0 = 1000\nI0 = 10\n").unwrap();

    // Minimal batch TOML with a 3-point sweep over beta.
    let batch_path = tmp.path().join("batch.toml");
    std::fs::write(&batch_path, format!(r#"
[config]
model = "{model}"
params = "{params}"
output_dir = "{out}"
seeds = {{ n = 1 }}
parallel = 1

[[scenario]]
name = "baseline"

[sweep]
beta = [0.2, 0.3, 0.4]
"#,
        model = golden_sir_basic().display(),
        params = params_path.display(),
        out = output.display(),
    )).unwrap();

    let st = Command::new(&bin)
        .args(["batch", "run", &batch_path.to_string_lossy()])
        .status().expect("spawn");
    assert!(st.success(), "batch sweep should succeed");

    // Find all run.json files (one per sweep point × scenario × seed = 3 total)
    let run_dirs: Vec<_> = walkdir(&output.join("sims")).into_iter()
        .filter(|p| p.join("run.json").exists()).collect();
    assert_eq!(run_dirs.len(), 3, "expected 3 runs for 3-point sweep");

    // The sweep point is a resolved value in the PARAMS level (not the
    // scenario delta, not the model hash). Its label encodes `beta=<v>`.
    let mut beta_values: Vec<f64> = Vec::new();
    for dir in &run_dirs {
        let meta = read_meta(dir);
        let label = level_label(&meta, "params"); // e.g. "beta=0.2"
        let beta: f64 = label.strip_prefix("beta=")
            .unwrap_or_else(|| panic!("params label should be beta=<v>, got {label}"))
            .parse().expect("beta value parses");
        beta_values.push(beta);
    }
    beta_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!((beta_values[0] - 0.2).abs() < 1e-9);
    assert!((beta_values[1] - 0.3).abs() < 1e-9);
    assert!((beta_values[2] - 0.4).abs() < 1e-9);

    // manifest.json must also carry sweep_point on each entry.
    // Manifest lives under sims/ after the 2026-04-19 unification.
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(output.join("sims").join("manifest.json")).unwrap()
    ).unwrap();
    let runs = manifest["runs"].as_array().expect("manifest.runs should be an array");
    assert_eq!(runs.len(), 3);
    for run in runs {
        assert!(run["sweep_point"]["beta"].is_number(),
            "manifest entry missing sweep_point.beta: {:?}", run);
    }

    // `camdl list` should show the beta values in PARAMS column
    let out = Command::new(&bin)
        .args(["list", &output.to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("beta=0.2"), "list should show beta=0.2: {}", stdout);
    assert!(stdout.contains("beta=0.3"), "list should show beta=0.3: {}", stdout);
    assert!(stdout.contains("beta=0.4"), "list should show beta=0.4: {}", stdout);
}

#[test]
fn simulate_batch_dry_run_prints_grid_no_output() {
    // --dry-run on `batch run` must print the resolved sweep grid
    // on stderr, exit 0, and touch zero files under output/runs/.
    let Some(bin) = skip_if_missing_binary() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");
    let params_path = tmp.path().join("params.toml");
    std::fs::write(&params_path, "beta = 0.3\ngamma = 0.1\nN0 = 1000\nI0 = 10\n").unwrap();
    let batch_path = tmp.path().join("batch.toml");
    std::fs::write(&batch_path, format!(r#"
[config]
model = "{model}"
params = "{params}"
output_dir = "{out}"
seeds = {{ n = 1 }}
parallel = 1

[[scenario]]
name = "baseline"

[sweep]
beta = [0.2, 0.3, 0.4]
"#,
        model = golden_sir_basic().display(),
        params = params_path.display(),
        out = output.display(),
    )).unwrap();

    let out = Command::new(&bin)
        .args(["batch", "run", &batch_path.to_string_lossy(), "--dry-run"])
        .output().expect("spawn");
    assert!(out.status.success(), "dry-run should exit 0");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("camdl batch run (dry run)"),
        "stderr should mark the dry run: {}", stderr);
    assert!(stderr.contains("Sweep grid"), "stderr should include sweep grid: {}", stderr);
    for beta in ["0.2", "0.3", "0.4"] {
        assert!(stderr.contains(beta), "stderr should include beta={}: {}", beta, stderr);
    }
    assert!(stderr.contains("no simulation"),
        "stderr should confirm no simulation ran: {}", stderr);

    // Must not have written any run files.
    let runs_dir = output.join("sims");
    assert!(!runs_dir.exists() ||
            walkdir(&runs_dir).into_iter().find(|p| p.join("run.json").exists()).is_none(),
        "dry-run must not write any run.json files");
}

#[test]
fn simulate_batch_flag_rejected_cleanly() {
    // `simulate FILE --batch OTHER` used to silently misinterpret the
    // first positional as the batch TOML path. With the flag removed,
    // the single-run parser errors cleanly on the unknown flag rather
    // than panicking or silently doing the wrong thing.
    let Some(bin) = skip_if_missing_binary() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let batch_path = tmp.path().join("foo.toml");
    std::fs::write(&batch_path, "").unwrap();

    let out = Command::new(&bin)
        .args(["simulate",
               &golden_sir_basic().to_string_lossy(),
               "--batch", &batch_path.to_string_lossy()])
        .output().expect("spawn");
    assert!(!out.status.success(), "`--batch` flag should fail cleanly, not run");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("unexpected argument") && stderr.contains("--batch"),
        "stderr should report unexpected argument, not panic: {}", stderr);
}

#[test]
fn show_prints_metadata() {
    let Some(bin) = skip_if_missing_binary() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");

    Command::new(&bin)
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline",
               "--seed", "42",
               "--cas",
               "--output-dir", &output.to_string_lossy(),
               "-o", &tmp.path().join("traj.tsv").to_string_lossy()])
        .status().expect("spawn");

    let dir = walkdir(&output.join("sims")).into_iter()
        .find(|p| p.join("run.json").exists()).unwrap();
    let meta = read_meta(&dir);
    let short = &meta["run_id"].as_str().unwrap()[..8];

    let out = Command::new(&bin)
        .args(["show", short, "--root", &output.to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success(), "show should succeed");
    let stdout = String::from_utf8_lossy(&out.stdout);
    // Check that show emits the key fields
    assert!(stdout.contains("baseline"), "should show scenario");
    assert!(stdout.contains("42"), "should show seed");
    assert!(stdout.contains("chain_binomial"), "should show backend (config level)");
}

/// Test (4a from review): `camdl show <fit-stage-hash>` resolves and
/// renders the FitStage payload. Pre-show-coverage-collapse, this
/// returned "unrecognised kind".
#[test]
fn show_renders_fit_stage_metadata() {
    let Some(bin) = skip_if_missing_binary() else { return };
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");
    let stage = output.join("fits/parent-abc12345/real/fit_42/scout");
    std::fs::create_dir_all(&stage).unwrap();
    let stage_run = r#"{
        "hash": "stage1234deadbeef0000000000000000000000000000000000000000stage1234",
        "version": "0.1.0+test",
        "created_at": "2026-04-30T12:00:00Z",
        "argv": ["camdl","fit","run","fit.toml","--stage","scout"],
        "status": {"completed": {"wall_time_seconds": 12.5}},
        "kind": {
            "kind": "fit-stage",
            "fit_hash": "abc12345deadbeef0000000000000000000000000000000000000000abc12345",
            "stage": "scout",
            "method": "if2",
            "backend": "chain_binomial",
            "seed": 42,
            "n_chains": 4,
            "best_loglik": -123.45,
            "best_chain": 1
        }
    }"#;
    std::fs::write(stage.join("run.json"), stage_run).unwrap();

    // Resolve by full path.
    let out = Command::new(&bin)
        .args(["show", &stage.to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success(), "show fit-stage failed: stderr={}",
        String::from_utf8_lossy(&out.stderr));
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("fit-stage"), "kind label missing: {}", s);
    assert!(s.contains("scout"),     "stage name missing: {}", s);
    assert!(s.contains("if2"),       "method missing: {}", s);
    assert!(s.contains("-123.45"),   "best_loglik missing: {}", s);
    assert!(s.contains("12.5"),      "wall time missing: {}", s);

    // Resolve by hash prefix.
    let out = Command::new(&bin)
        .args(["show", "stage1234", "--root", &output.to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success(), "show by stage-hash prefix failed: stderr={}",
        String::from_utf8_lossy(&out.stderr));
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("scout"));
}

/// Test (4b from review): `camdl show <profile-leaf-hash>` for the
/// per-seed Profile leaf inside a ReplicateSet umbrella.
#[test]
fn show_renders_profile_leaf_metadata() {
    let Some(bin) = skip_if_missing_binary() else { return };
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");
    let umbrella = output.join("profiles/foo-aaaa1111");
    let leaf = umbrella.join("replicates/seed_42");
    std::fs::create_dir_all(&leaf).unwrap();

    // Umbrella: ReplicateSet
    let umbrella_run = r#"{
        "hash": "aaaa1111deadbeef0000000000000000000000000000000000000000aaaa1111",
        "version": "0.1.0+test",
        "created_at": "2026-04-30T12:00:00Z",
        "argv": ["camdl","profile","--seeds","42"],
        "status": {"completed": {"wall_time_seconds": 25.0}},
        "kind": {
            "kind": "replicate-set",
            "dim_name": "seed",
            "keys": ["seed_42"],
            "child_kind": "profile",
            "inner_content_hash": "ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000"
        }
    }"#;
    std::fs::write(umbrella.join("run.json"), umbrella_run).unwrap();

    // Leaf: Profile
    let leaf_run = r#"{
        "hash": "leaf1111deadbeef0000000000000000000000000000000000000000leaf1111",
        "version": "0.1.0+test",
        "created_at": "2026-04-30T12:00:00Z",
        "argv": ["camdl","profile","--seeds","42"],
        "status": {"completed": {"wall_time_seconds": 12.5}},
        "kind": {
            "kind": "profile",
            "model": "sir.camdl",
            "model_hash": "ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000",
            "focal_params": ["beta", "gamma"],
            "grid": [
                {"param": "beta",  "values": [1.0, 2.0, 3.0]},
                {"param": "gamma", "values": [0.1, 0.2]}
            ],
            "n_starts": 3,
            "if2_config_hash": "1111111111111111111111111111111111111111111111111111111111111111",
            "base_params_hash": "2222222222222222222222222222222222222222222222222222222222222222",
            "seed_base": 42,
            "total_jobs": 18
        }
    }"#;
    std::fs::write(leaf.join("run.json"), leaf_run).unwrap();

    let out = Command::new(&bin)
        .args(["show", "leaf1111", "--root", &output.to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success(), "show profile-leaf failed: stderr={}",
        String::from_utf8_lossy(&out.stderr));
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("profile"),    "kind label missing: {}", s);
    assert!(s.contains("beta, gamma"),"focal params missing: {}", s);
    assert!(s.contains("3 per grid point"), "n_starts missing: {}", s);
    assert!(s.contains("18"),         "total_jobs missing: {}", s);
    assert!(s.contains("12.5"),       "wall time missing: {}", s);
}

/// Test (3 from review): single-seed profile umbrella layout.
/// Asserts the on-disk shape via handcrafted run.json files —
/// `discover_profiles` must accept the umbrella, find the leaf,
/// and `camdl list --kind profile` must surface it.
#[test]
fn single_seed_profile_layout_lists_via_discover_profiles() {
    let Some(bin) = skip_if_missing_binary() else { return };
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");
    let umbrella = output.join("profiles/sir-cccc3333");
    let leaf = umbrella.join("replicates/seed_42");
    std::fs::create_dir_all(&leaf).unwrap();

    // Umbrella: ReplicateSet with one seed_42 child.
    let umbrella_run = r#"{
        "hash": "cccc3333deadbeef0000000000000000000000000000000000000000cccc3333",
        "version": "0.1.0+test",
        "created_at": "2026-04-30T12:00:00Z",
        "argv": ["camdl","profile","--seed","42"],
        "status": {"completed": {"wall_time_seconds": 5.0}},
        "kind": {
            "kind": "replicate-set",
            "dim_name": "seed",
            "keys": ["seed_42"],
            "child_kind": "profile",
            "inner_content_hash": "ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000"
        }
    }"#;
    std::fs::write(umbrella.join("run.json"), umbrella_run).unwrap();

    // Leaf: Profile.
    let leaf_run = r#"{
        "hash": "leaf3333deadbeef0000000000000000000000000000000000000000leaf3333",
        "version": "0.1.0+test",
        "created_at": "2026-04-30T12:00:00Z",
        "argv": ["camdl","profile","--seed","42"],
        "status": {"completed": {"wall_time_seconds": 5.0}},
        "kind": {
            "kind": "profile",
            "model": "sir.camdl",
            "model_hash": "ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000",
            "focal_params": ["beta"],
            "grid": [{"param": "beta", "values": [1.0, 2.0]}],
            "n_starts": 1,
            "if2_config_hash": "1111111111111111111111111111111111111111111111111111111111111111",
            "base_params_hash": "2222222222222222222222222222222222222222222222222222222222222222",
            "seed_base": 42,
            "total_jobs": 2
        }
    }"#;
    std::fs::write(leaf.join("run.json"), leaf_run).unwrap();

    // `camdl list --kind profile` must surface the umbrella.
    let out = Command::new(&bin)
        .args(["list", &output.to_string_lossy(), "--kind", "profile"])
        .output().expect("spawn");
    assert!(out.status.success(), "list profile failed: stderr={}",
        String::from_utf8_lossy(&out.stderr));
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("sir-cccc3333"),
        "list must show the profile dir: {}", s);
    assert!(s.contains("beta"),
        "list must show the focal param: {}", s);

    // `camdl show <umbrella_hash>` must recognize the kind.
    let out = Command::new(&bin)
        .args(["show", "cccc3333", "--root", &output.to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success(), "show on umbrella hash failed: stderr={}",
        String::from_utf8_lossy(&out.stderr));
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("replicate-set"),
        "show must label kind: {}", s);
    assert!(s.contains("seed_42"),
        "show must list child key: {}", s);
}

/// Test (5 from review): `camdl label` should work uniformly across
/// run kinds, not just fits. Pre-rename, this only worked on fits.
#[test]
fn label_works_on_sim_and_profile_runs() {
    let Some(bin) = skip_if_missing_binary() else { return };
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");

    // Sim run — golden_sir_basic has no parameter defaults, so pass them.
    Command::new(&bin)
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--param", "beta=2.0",
               "--param", "gamma=0.3",
               "--param", "N0=1000",
               "--param", "I0=10",
               "--seed", "1", "--cas",
               "--output-dir", &output.to_string_lossy(),
               "-o", &tmp.path().join("traj.tsv").to_string_lossy()])
        .status().expect("spawn");

    let sim_dir = walkdir(&output.join("sims")).into_iter()
        .find(|p| p.join("run.json").exists()).expect("one sim dir");
    let sim_meta = read_meta(&sim_dir);
    let sim_hash: String = sim_meta["run_id"].as_str().unwrap().chars().take(8).collect();

    // Label the sim by run_id prefix.
    let status = Command::new(&bin)
        .args(["label", &sim_hash, "test sim label",
               "--root", &output.to_string_lossy()])
        .status().expect("spawn");
    assert!(status.success(), "label on sim must succeed");

    // Re-read; the label persists in the RunRecord provenance.
    let sim_meta2 = read_meta(&sim_dir);
    assert_eq!(sim_meta2["provenance"]["label"].as_str(), Some("test sim label"),
        "sim label must persist on RunRecord.provenance.label. got: {:?}",
        sim_meta2);

    // Now plant a profile umbrella + leaf and label the umbrella by hash.
    let umbrella = output.join("profiles/p-bbbb2222");
    let leaf = umbrella.join("replicates/seed_1");
    std::fs::create_dir_all(&leaf).unwrap();
    let umbrella_run = r#"{
        "hash": "bbbb2222deadbeef0000000000000000000000000000000000000000bbbb2222",
        "version": "0.1.0+test",
        "created_at": "2026-04-30T12:00:00Z",
        "argv": [],
        "status": {"completed": {"wall_time_seconds": 1.0}},
        "kind": {
            "kind": "replicate-set",
            "dim_name": "seed",
            "keys": ["seed_1"],
            "child_kind": "profile",
            "inner_content_hash": "ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000"
        }
    }"#;
    std::fs::write(umbrella.join("run.json"), umbrella_run).unwrap();

    let status = Command::new(&bin)
        .args(["label", "bbbb2222", "test profile label",
               "--root", &output.to_string_lossy()])
        .status().expect("spawn");
    assert!(status.success(), "label on profile must succeed");

    let umbrella_meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(umbrella.join("run.json")).unwrap()
    ).unwrap();
    assert_eq!(umbrella_meta["label"].as_str(), Some("test profile label"),
        "profile-umbrella label must persist on Run.label. got: {:?}",
        umbrella_meta);
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Collect all directory paths under `root` (non-recursive children of
/// each level; bounded depth is fine for our 3-level CAS layout).
/// Every directory at any depth under `root`. Callers filter to leaves with
/// `.filter(|p| p.join("run.json").exists())`. Depth-agnostic so it handles
/// the factored 5-level CAS path (model/config/params/scenario/seed), not
/// just the legacy 3-level layout.
fn walkdir(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue; };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                out.push(p.clone());
                stack.push(p);
            }
        }
    }
    out
}
