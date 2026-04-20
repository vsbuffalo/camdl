//! End-to-end tests for `camdl simulate --cas` and `camdl list/show/cat`.
//!
//! These shell out to the built `camdl-sim` binary in `target/release/`
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
        .join("../../target/release/camdl-sim");
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
        eprintln!("skipping: camdl-sim binary not built at {}", bin.display());
        return None;
    }
    Some(bin)
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

    // run.json should have full metadata including argv + version
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("run.json")).unwrap()
    ).unwrap();
    // Unified Run schema nests simulate-specific fields inside `kind`;
    // shared Run fields stay at the top.
    assert_eq!(meta["kind"]["seed"], 42);
    assert_eq!(meta["kind"]["scenario"], "baseline");
    assert_eq!(meta["kind"]["kind"], "simulate");
    assert!(meta["kind"]["sim_hash"].as_str().unwrap().len() == 64);
    assert!(meta["argv"].as_array().unwrap().len() >= 4);
    assert!(meta["version"].as_str().unwrap().contains("+"),
        "version should include git hash suffix");
    assert!(meta["hash"].as_str().unwrap().len() == 64);
    assert!(meta["created_at"].as_str().is_some());
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

    let seeds: Vec<String> = dirs.iter()
        .map(|d| d.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert!(seeds.iter().any(|n| n == "seed_42"));
    assert!(seeds.iter().any(|n| n == "seed_43"));
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
    assert!(stderr.contains("simulate batch"),
        "error should hint at `simulate batch`: {}", stderr);
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

/// `--starts-from <hash>` should resolve a short-hash prefix to the
/// matching fit-stage directory. Hardening #9.
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
        "argv": [],"wall_time_seconds": 1.0,
        "kind": {{"kind":"fit-stage","fit_hash":"f","stage":"scout",
                 "method":"if2","seed":1,"n_chains":2}}
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
method = "if2"
chains = 2
particles = 50
iterations = 3
cooling = 0.7
starts_from = "{{use CLI}}"
"#, ir.display(), data.display())).unwrap();

    let out = Command::new(&bin)
        .current_dir(tmp.path())
        .args(["fit", "run", &fit_toml.to_string_lossy(),
               "--stage", "refine",
               "--starts-from", "zzzznonexistent"])
        .output().expect("spawn");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "bad hash must fail");
    assert!(stderr.contains("no fit stage matching hash prefix"),
        "expected 'no fit stage matching hash prefix', got: {}", stderr);

    // Good hash: resolves to the fake stage we planted. Resolution
    // happens before the fit actually does anything expensive, so
    // verifying the success path means checking that we get past
    // arg parsing — the fit itself may still fail downstream (the
    // model IR is empty), but the --starts-from lookup succeeded.
    // We check that the stderr does NOT contain the 'no fit stage
    // matching' message.
    let out = Command::new(&bin)
        .current_dir(tmp.path())
        .args(["fit", "run", &fit_toml.to_string_lossy(),
               "--stage", "refine",
               "--starts-from", "deadbeef"])
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
    let ir = tmp.path().join("dummy.ir.json");
    // A minimal-but-valid IR so FitConfigV2 can load and hash the
    // referenced file. Actual content doesn't matter — where doesn't
    // run anything.
    std::fs::write(&ir, r#"{"compartments":[],"parameters":[]}"#).unwrap();
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
beta = {{ bounds = [0.01, 2.0] }}

[fixed]
N0 = 1000

[stages.mle]
method = "if2"
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
/// should hide fit rows. Covers cleanup.md:L2.
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
        "argv":["camdl","fit","run","demo.toml"],"wall_time_seconds":1.0,
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
        "wall_time_seconds": 3.2,
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
    // Locate the run dir, corrupt the stored hash.
    let dir = walkdir(&output.join("sims")).into_iter()
        .find(|p| p.join("run.json").exists()).expect("one run");
    let content = std::fs::read_to_string(dir.join("run.json")).unwrap();
    let mut v: serde_json::Value = serde_json::from_str(&content).unwrap();
    v["hash"] = serde_json::Value::String("0".repeat(64));
    std::fs::write(dir.join("run.json"), serde_json::to_string_pretty(&v).unwrap()).unwrap();

    // Second run should detect the stale hash and warn.
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
        "argv":["camdl","fit","run","demo.toml"],"wall_time_seconds":1.0,
        "kind":{"kind":"fit","model":"demo.camdl","model_hash":"m",
        "fit_toml_path":"demo.toml","fit_toml_hash":"h",
        "data_hashes":{},"estimated":["beta"],"fixed":{},"stages_declared":["mle"]}
    }"#;
    std::fs::write(fit_dir.join("run.json"), run_json).unwrap();
    let out = Command::new(&bin)
        .args(["show", "deadbee", &output.to_string_lossy()])
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
        "wall_time_seconds": 3.2,
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

    // Find the cached dir, derive short hash
    let dir = walkdir(&output.join("sims")).into_iter()
        .find(|p| p.join("run.json").exists()).unwrap();
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("run.json")).unwrap()
    ).unwrap();
    let sim_hash_full = meta["kind"]["sim_hash"].as_str().unwrap();
    let short = &sim_hash_full[..8];

    // `camdl cat <short>` uniquely resolves and emits the TSV
    let out = Command::new(&bin)
        .args(["cat", short, &output.to_string_lossy()])
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
        .args(["simulate", "batch", &batch_path.to_string_lossy()])
        .status().expect("spawn");
    assert!(st.success(), "batch sweep should succeed");

    // Find all run.json files (one per sweep point × scenario × seed = 3 total)
    let run_dirs: Vec<_> = walkdir(&output.join("sims")).into_iter()
        .filter(|p| p.join("run.json").exists()).collect();
    assert_eq!(run_dirs.len(), 3, "expected 3 runs for 3-point sweep");

    // Each run.json must have sweep_point with the beta value
    let mut beta_values: Vec<f64> = Vec::new();
    for dir in &run_dirs {
        let meta: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("run.json")).unwrap()
        ).unwrap();
        let sp = &meta["kind"]["sweep_point"];
        assert!(!sp.is_null(), "run.json must have kind.sweep_point: {:?}", meta);
        let beta = sp["beta"].as_f64().expect("sweep_point.beta must be a number");
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
    // --dry-run on `simulate batch` must print the resolved sweep grid
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
        .args(["simulate", "batch", &batch_path.to_string_lossy(), "--dry-run"])
        .output().expect("spawn");
    assert!(out.status.success(), "dry-run should exit 0");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("camdl simulate batch (dry run)"),
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
            walkdir(&runs_dir).into_iter().filter(|p| p.join("run.json").exists()).next().is_none(),
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
    assert!(stderr.contains("unknown flag") && stderr.contains("--batch"),
        "stderr should report unknown flag, not panic: {}", stderr);
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
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("run.json")).unwrap()
    ).unwrap();
    let short = &meta["kind"]["sim_hash"].as_str().unwrap()[..8];

    let out = Command::new(&bin)
        .args(["show", short, &output.to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success(), "show should succeed");
    let stdout = String::from_utf8_lossy(&out.stdout);
    // Check that show emits the key fields
    assert!(stdout.contains("baseline"), "should show scenario");
    assert!(stdout.contains("42"), "should show seed");
    assert!(stdout.contains("chain_binomial"), "should show backend");
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Collect all directory paths under `root` (non-recursive children of
/// each level; bounded depth is fine for our 3-level CAS layout).
fn walkdir(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if !root.exists() { return out; }
    let Ok(entries) = std::fs::read_dir(root) else { return out; };
    for e in entries.flatten() {
        let p = e.path();
        if !p.is_dir() { continue; }
        // Recurse up to ~3 levels: sim_hash / scen-hash / seed_n
        if let Ok(entries2) = std::fs::read_dir(&p) {
            for e2 in entries2.flatten() {
                let p2 = e2.path();
                if !p2.is_dir() { continue; }
                if let Ok(entries3) = std::fs::read_dir(&p2) {
                    for e3 in entries3.flatten() {
                        let p3 = e3.path();
                        if p3.is_dir() { out.push(p3); }
                    }
                }
            }
        }
    }
    out
}
