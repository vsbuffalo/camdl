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

fn skip_if_missing_binary() -> PathBuf {
    let bin = binary();
    assert!(
        bin.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test` (gh#105)",
        bin.display()
    );
    bin
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
    let bin = skip_if_missing_binary();
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
fn cas_second_identical_run_does_not_rewrite_leaf() {
    // CAS is the default. A second, byte-identical run re-simulates (the
    // trajectory is needed for the -o mirror) but the store commit is
    // idempotent: `commit_atomic` resolves the existing identical leaf to
    // `AlreadyCompleted` and never re-writes its `traj.tsv`. The proof is the
    // unchanged mtime.
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");

    let run_once = || {
        Command::new(&bin)
            .args(["simulate", &golden_sir_basic().to_string_lossy(),
                   "--scenario", "baseline",
                   "--seed", "42",
                   "--output-dir", &output.to_string_lossy(),
                   "-o", &tmp.path().join("traj.tsv").to_string_lossy()])
            .output()
            .expect("spawn")
    };

    let first = run_once();
    assert!(first.status.success());
    let stderr1 = String::from_utf8_lossy(&first.stderr);
    assert!(stderr1.contains("stored"), "first run stderr should report the run was stored: {}", stderr1);

    // Wait long enough that the filesystem mtime would differ if rewritten.
    let cache_path = walkdir(&output.join("sims")).into_iter()
        .find(|p| p.join("traj.tsv").exists()).unwrap()
        .join("traj.tsv");
    let mtime1 = std::fs::metadata(&cache_path).unwrap().modified().unwrap();

    std::thread::sleep(std::time::Duration::from_millis(10));

    let second = run_once();
    assert!(second.status.success());

    // mtime unchanged — the idempotent commit must not re-write the leaf.
    let mtime2 = std::fs::metadata(&cache_path).unwrap().modified().unwrap();
    assert_eq!(mtime1, mtime2, "idempotent commit must not overwrite traj.tsv");
}

#[test]
fn cas_different_seed_new_cache_entry() {
    let bin = skip_if_missing_binary();
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
/// params/backend/dt/seed must NOT collide to one CAS entry: the runid
/// `model`-level digest (the whole-IR structural hash) has to separate them.
///
/// The two models share a basename (`model.ir.json`) so the path's
/// model-stem prefix is identical — the ONLY thing that can separate
/// them is the model-level hash. They differ only in the `recovery`
/// transition's rate (×2), a structural field folded into the model
/// digest but NOT a parameter value, so `base_params_canonical` is
/// identical between them.
#[test]
fn cas_different_models_do_not_collide() {
    let bin = skip_if_missing_binary();
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
fn cas_multi_seeds_writes_one_leaf_per_seed() {
    // `--cas supports single runs only` is gone: CAS is the default and
    // multi-run writes one content-addressed leaf per cell — exactly like
    // `batch run`. `--seeds 1:3` is three seed-slots → three leaves with
    // three distinct seed-level segments.
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");

    let out = Command::new(&bin)
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline",
               "--seeds", "1:3",
               "--output-dir", &output.to_string_lossy()])
        .output()
        .expect("spawn");
    assert!(out.status.success(), "multi-seed simulate should succeed: {}",
        String::from_utf8_lossy(&out.stderr));

    let dirs: Vec<_> = walkdir(&output.join("sims")).into_iter()
        .filter(|p| p.join("run.json").exists()).collect();
    assert_eq!(dirs.len(), 3, "three seeds must produce three leaves");

    // Three distinct run_ids (the seed is in the key — count-in-the-key).
    let mut run_ids: Vec<String> = dirs.iter()
        .map(|d| read_meta(d)["run_id"].as_str().unwrap().to_string())
        .collect();
    run_ids.sort();
    run_ids.dedup();
    assert_eq!(run_ids.len(), 3, "three seeds must yield three distinct run_ids");

    // The seed labels cover 1, 2, 3 (explicit --seeds are used verbatim).
    let labels: Vec<String> = dirs.iter()
        .map(|d| level_label(&read_meta(d), "seed").to_string()).collect();
    for s in ["seed_1", "seed_2", "seed_3"] {
        assert!(labels.iter().any(|l| l == s), "missing {s} in {labels:?}");
    }
}

/// Count-in-the-key: `--replicates N` (no `--draws`) writes N distinct leaves,
/// one per stochastic replicate, each with its own XOR-mixed seed and a
/// distinct `run_id`. A lone run writes exactly one leaf. This guards the
/// engine wiring that maps `--replicates` onto the rep dimension for the
/// `Point` param-source (regression: it previously collapsed to one run).
#[test]
fn cas_replicates_write_one_leaf_each_and_single_run_writes_one() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();

    // 3 replicates → 3 leaves, 3 distinct run_ids.
    let out3 = tmp.path().join("out3");
    let st = Command::new(&bin)
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline", "--seed", "42", "--replicates", "3",
               "--output-dir", &out3.to_string_lossy(),
               "-o", &tmp.path().join("c3.tsv").to_string_lossy()])
        .status().expect("spawn");
    assert!(st.success(), "3-replicate simulate should succeed");
    let dirs3: Vec<_> = walkdir(&out3.join("sims")).into_iter()
        .filter(|p| p.join("run.json").exists()).collect();
    assert_eq!(dirs3.len(), 3, "3 replicates must write 3 leaves");
    let mut ids3: Vec<String> = dirs3.iter()
        .map(|d| read_meta(d)["run_id"].as_str().unwrap().to_string()).collect();
    ids3.sort(); ids3.dedup();
    assert_eq!(ids3.len(), 3, "3 replicates must have 3 distinct run_ids (seed in key)");

    // 1 run (no --replicates) → exactly one leaf.
    let out1 = tmp.path().join("out1");
    let st = Command::new(&bin)
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline", "--seed", "42",
               "--output-dir", &out1.to_string_lossy(),
               "-o", &tmp.path().join("c1.tsv").to_string_lossy()])
        .status().expect("spawn");
    assert!(st.success(), "single-run simulate should succeed");
    let dirs1: Vec<_> = walkdir(&out1.join("sims")).into_iter()
        .filter(|p| p.join("run.json").exists()).collect();
    assert_eq!(dirs1.len(), 1, "a lone simulate run must write exactly one leaf");
}

/// The load-bearing invariant: `simulate --seeds …` and `batch run` write
/// leaves at the SAME store paths with the SAME `run_id`s and BYTE-IDENTICAL
/// `traj.tsv` for the same (model, config, params, scenario, process_seed)
/// cells. Both go through the shared `CasSink` write path, so divergence here
/// means the two entry points resolved identity differently — a silent-wrong-
/// answer class bug (a `simulate` leaf would never be served to a `batch`
/// cache lookup, or vice versa).
#[test]
fn simulate_and_batch_leaves_are_byte_identical() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();

    let params = tmp.path().join("params.toml");
    std::fs::write(&params, "beta = 0.3\ngamma = 0.1\nN0 = 1000\nI0 = 10\n").unwrap();

    let sim_out = tmp.path().join("sim_out");
    let batch_out = tmp.path().join("batch_out");

    // simulate over an explicit seed list.
    let st = Command::new(&bin)
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline",
               "--params", &params.to_string_lossy(),
               "--seeds", "10,11,12",
               "--output-dir", &sim_out.to_string_lossy(),
               "-o", &tmp.path().join("c.tsv").to_string_lossy()])
        .status().expect("spawn");
    assert!(st.success(), "simulate --seeds should succeed");

    // The equivalent batch run: same model, params, seed list, baseline.
    let batch_toml = tmp.path().join("b.toml");
    std::fs::write(&batch_toml, format!(r#"
[config]
model = "{model}"
params = "{params}"
output_dir = "{out}"
seeds = {{ list = [10, 11, 12] }}
parallel = 1

[[scenario]]
name = "baseline"
"#,
        model = golden_sir_basic().display(),
        params = params.display(),
        out = batch_out.display(),
    )).unwrap();
    let st = Command::new(&bin)
        .args(["batch", "run", &batch_toml.to_string_lossy()])
        .status().expect("spawn");
    assert!(st.success(), "batch run should succeed");

    // Enumerate the simulate leaves; for each, the same store-relative path
    // must exist under batch with the same run_id and identical traj bytes.
    let sims_root = sim_out.join("sims");
    let sim_leaves: Vec<_> = walkdir(&sims_root).into_iter()
        .filter(|p| p.join("run.json").exists()).collect();
    assert_eq!(sim_leaves.len(), 3, "expected 3 simulate leaves");

    for sl in &sim_leaves {
        let rel = sl.strip_prefix(&sims_root).unwrap();
        let bl = batch_out.join("sims").join(rel);
        assert!(bl.join("run.json").exists(),
            "batch missing the same-path leaf: {}", rel.display());

        let sid = read_meta(sl)["run_id"].as_str().unwrap().to_string();
        let bid = read_meta(&bl)["run_id"].as_str().unwrap().to_string();
        assert_eq!(sid, bid, "run_id mismatch at {}", rel.display());

        let sbytes = std::fs::read(sl.join("traj.tsv")).unwrap();
        let bbytes = std::fs::read(bl.join("traj.tsv")).unwrap();
        assert_eq!(sbytes, bbytes,
            "traj.tsv bytes differ between simulate and batch at {}", rel.display());
    }
}

#[test]
fn list_shows_cached_runs() {
    let bin = skip_if_missing_binary();
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
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let results = tmp.path().join("results");
    // Fake a stage dir with a known-hash run.json. The resolver
    // walks results/fits/**, so we need to place the run.json
    // under that structure.
    let stage = results.join("fits").join("demo-abc12345")
        .join("real").join("fit_1").join("scout");
    std::fs::create_dir_all(&stage).unwrap();
    // gh#147: the resolver matches a `FitStage` `runid::RunRecord` leaf on its
    // `run_id` hex prefix. Plant a leaf whose `run_id` starts `deadbeef`.
    let target_hash = "deadbeefc0ffee00000000000000000000000000000000000000000000000000";
    let run_json = format!(r#"{{
        "format_version":1,"kind":"fit_stage","run_id":"{}","hash_version":1,
        "ir_version":"0.7","engine_version":"0.1.0+test",
        "levels":[
            {{"name":"fit","label":"demo","hash":"abc1234500000000000000000000000000000000000000000000000000000000","schema_version":1}},
            {{"name":"stage","label":"01-scout","hash":"1fb03eee00000000000000000000000000000000000000000000000000000000","schema_version":1}},
            {{"name":"seed","label":"seed_1","hash":"06cbd6b300000000000000000000000000000000000000000000000000000000","schema_version":1}}
        ],
        "status":"completed","artifacts":{{}},
        "inputs":{{"stage":"scout","method":"if2","backend":"chain_binomial","seed":1,"n_chains":2}},
        "provenance":{{"created_at":"2026-04-19T12:00:00Z","argv":[]}}
    }}"#, target_hash);
    std::fs::write(stage.join("run.json"), run_json).unwrap();

    // Exercise the short-hash stage resolver through `fit run`: the removed
    // `--starts-from` flag must error with the actionable replacement
    // message, and the new `--init from_mle --mle <hash>` spelling resolves
    // the same planted stage. Run from `tmp` so the default `./results`
    // resolver finds the leaf planted above.
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

/// `camdl list --kind fit` should hide sim rows entirely; `--kind sim`
/// should hide fit rows. Covers docs/dev/notes/2026-04-20-unified-output-tree-cleanup.md:L2.
#[test]
fn list_kind_filter_isolates_sections() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");

    // Cache one sim.
    Command::new(&bin)
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline", "--seed", "7", "--cas",
               "--output-dir", &output.to_string_lossy(),
               "-o", &tmp.path().join("t.tsv").to_string_lossy()])
        .status().expect("spawn");
    // Synthesise a CAS fit (single `mle` stage leaf + sidecar).
    write_cas_fit(&output, "demo", "abc12345", &["mle"], "m");

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
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");
    // gh#147 (M3.2): a CAS fit segment with two stage leaves + the fit-level
    // sidecar. `read_fit_segment` derives one fit entry whose `stages_declared`
    // comes from the leaves (`scout`, `refine`).
    write_cas_fit(&output, "demo", "abc12345", &["scout", "refine"], "m000");

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

/// Tamper-with-artifact regression: a corrupted `traj.tsv` whose bytes no
/// longer match the leaf's exact-set manifest must never be served. With CAS
/// the default, a re-run always re-simulates and commits idempotently; the
/// commit's claim resolution classifies the tampered leaf as `Stale` and
/// repairs it (the staging leaf is renamed into place), so the corrupt bytes
/// are replaced with the correct trajectory.
#[test]
fn cas_tampered_leaf_is_repaired_on_rerun() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");

    let run_once = || {
        Command::new(&bin)
            .args(["simulate", &golden_sir_basic().to_string_lossy(),
                   "--scenario", "baseline",
                   "--seed", "42",
                   "--output-dir", &output.to_string_lossy(),
                   "-o", &tmp.path().join("traj.tsv").to_string_lossy()])
            .output().expect("spawn")
    };

    let first = run_once();
    assert!(first.status.success());
    let dir = walkdir(&output.join("sims")).into_iter()
        .find(|p| p.join("run.json").exists()).expect("one run");
    let good = std::fs::read(dir.join("traj.tsv")).unwrap();

    // Corrupt the cached trajectory.
    std::fs::write(dir.join("traj.tsv"), b"tampered-and-shorter").unwrap();

    // The re-run repairs it: the leaf's traj.tsv is back to the correct bytes,
    // never left holding the corrupt content.
    let out = run_once();
    assert!(out.status.success(), "re-run should succeed");
    let repaired = walkdir(&output.join("sims")).into_iter()
        .find(|p| p.join("run.json").exists()).expect("one run");
    let after = std::fs::read(repaired.join("traj.tsv")).unwrap();
    assert_ne!(after, b"tampered-and-shorter",
        "tampered bytes must not survive a re-run");
    assert_eq!(after, good, "re-run must restore the correct trajectory bytes");
}

#[test]
fn show_resolves_fit_by_hash_prefix() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");
    // gh#147 (M3.2): a CAS fit is addressed by its stage-leaf `run_id` prefix;
    // there is no legacy fit-wide record. `run_id` starts `deadbeef` so the
    // short prefix matches.
    let run_id = "deadbeefc0ffee00000000000000000000000000000000000000000000000000";
    write_cas_fit_stage(&output, run_id, "demo");
    let out = Command::new(&bin)
        .args(["show", "deadbee", "--root", &output.to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success(), "show by run_id prefix should resolve: stderr={}",
        String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("fit_stage"), "should render a fit-stage: {stdout}");
    assert!(stdout.contains("scout"), "stage label missing: {stdout}");
}


#[test]
fn cat_emits_cached_trajectory() {
    let bin = skip_if_missing_binary();
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
    // The canonical leaf carries no `# camdl <version>` provenance header
    // (that lives in run.json's `engine_version`); the trajectory TSV is the
    // batch-converged form starting with the `t` column header.
    assert!(stdout.starts_with("t\t"), "cat should emit the trajectory header row: {:?}",
        &stdout[..stdout.len().min(40)]);
    assert!(stdout.contains("\tS\t") || stdout.contains("\tI\t"),
        "cat should include compartment columns");

    // Cached trajectory bytes should match the stdout of `cat`
    let cached = std::fs::read(dir.join("traj.tsv")).unwrap();
    assert_eq!(out.stdout, cached, "cat output must match cached bytes byte-for-byte");
}

#[test]
fn batch_sweep_records_sweep_point_in_run_json() {
    // Regression: before this fix, batch sweeps wrote run.json with no
    // record of the sweep parameter values — you could see there were 8
    // distinct scen_hashes but not which beta value produced which
    // trajectory. This test runs a minimal --batch with a 3-point sweep and
    // asserts the sweep point is recoverable from the live tree (the `params`
    // level label + `camdl list`). manifest.json was retired (gh#147 item D);
    // the per-leaf run.json + derived index.json are the only index.
    let bin = skip_if_missing_binary();
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

    // manifest.json is retired (gh#147 item D) — `batch run` must NOT write
    // a batch-level index. The live tree (run.json per leaf) is the truth.
    assert!(!output.join("sims").join("manifest.json").exists(),
        "batch run must not write sims/manifest.json (retired)");

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
    let bin = skip_if_missing_binary();
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
    let bin = skip_if_missing_binary();
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
    let bin = skip_if_missing_binary();
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
/// A content-addressed fit-stage leaf (`runid::RunRecord`, M3.2) at
/// `fits/{fit}/{NN-stage}/{seed}/run.json`, addressable by `run_id` prefix.
fn write_cas_fit_stage(output: &Path, run_id: &str, fit_label: &str) -> PathBuf {
    let leaf = output.join("fits")
        .join(format!("{fit_label}-5ca1ab1e"))
        .join("01-scout-1fb03eee")
        .join("seed_42-06cbd6b3");
    std::fs::create_dir_all(&leaf).unwrap();
    let rec = format!(r#"{{
        "format_version": 1,
        "kind": "fit_stage",
        "run_id": "{run_id}",
        "hash_version": 1,
        "ir_version": "0.7",
        "engine_version": "0.1.0+test",
        "levels": [
            {{"name":"fit","label":"{fit_label}","hash":"5ca1ab1e00000000000000000000000000000000000000000000000000000000","schema_version":1}},
            {{"name":"stage","label":"01-scout","hash":"1fb03eee00000000000000000000000000000000000000000000000000000000","schema_version":1}},
            {{"name":"seed","label":"seed_42","hash":"06cbd6b300000000000000000000000000000000000000000000000000000000","schema_version":1}}
        ],
        "status": "completed",
        "artifacts": {{}},
        "inputs": {{"stage":"scout","method":"if2","backend":"chain_binomial","seed":42,"n_chains":4,"best_loglik":-123.45,"best_chain":1}},
        "provenance": {{"created_at":"2026-04-30T12:00:00Z","argv":["camdl","fit","run"]}}
    }}"#);
    std::fs::write(leaf.join("run.json"), rec).unwrap();
    leaf
}

/// Write a content-addressed fit segment `fits/<label>-<fit_h8>/` with one
/// `FitStage` leaf per stage (`<NN>-<stage>-<h8>/seed_1-<h8>/run.json`) plus the
/// fit-level sidecar (`fit.meta.json` — the label + model-identity home). This is
/// the shape `read_fit_segment` derives a single fit-level entry from, so
/// `list` / `fit table` see one fit with `stages_declared` taken from the
/// leaves. Returns the segment dir.
fn write_cas_fit(output: &Path, label: &str, fit_h8: &str, stages: &[&str], model_identity: &str) -> PathBuf {
    let seg = output.join("fits").join(format!("{label}-{fit_h8}"));
    std::fs::create_dir_all(&seg).unwrap();
    let fit_hash = format!("{fit_h8}{}", "0".repeat(64 - fit_h8.len()));
    for (i, stage) in stages.iter().enumerate() {
        let nn = i + 1;
        let leaf = seg
            .join(format!("{nn:02}-{stage}-1fb03eee"))
            .join("seed_1-06cbd6b3");
        std::fs::create_dir_all(&leaf).unwrap();
        // Any distinct 64-hex run_id per leaf; `read_fit_segment` reads the
        // `fit`-level hash (shared) for the fit-level entry's `run.hash`.
        let run_id = format!("{:0<64}", format!("abc{nn}"));
        let rec = format!(r#"{{
            "format_version": 1,
            "kind": "fit_stage",
            "run_id": "{run_id}",
            "hash_version": 1,
            "ir_version": "0.7",
            "engine_version": "0.1.0+test",
            "levels": [
                {{"name":"fit","label":"{label}","hash":"{fit_hash}","schema_version":1}},
                {{"name":"stage","label":"{nn:02}-{stage}","hash":"1fb03eee00000000000000000000000000000000000000000000000000000000","schema_version":1}},
                {{"name":"seed","label":"seed_1","hash":"06cbd6b300000000000000000000000000000000000000000000000000000000","schema_version":1}}
            ],
            "status": "completed",
            "artifacts": {{}},
            "inputs": {{"stage":"{stage}","method":"if2","backend":"chain_binomial","seed":1,"n_chains":2}},
            "provenance": {{"created_at":"2026-04-19T12:00:00Z","argv":["camdl","fit","run"]}}
        }}"#);
        std::fs::write(leaf.join("run.json"), rec).unwrap();
    }
    std::fs::write(
        seg.join("fit.meta.json"),
        format!(r#"{{"model_identity":"{model_identity}","model_path":"demo.camdl","fit_toml_path":"demo.toml"}}"#),
    )
    .unwrap();
    seg
}

/// gh#147 (M3.2): `camdl show <fit-stage path | run_id prefix>` renders the
/// CAS fit-stage `RunRecord` — factored levels + the recorded FitStageMeta.
#[test]
fn show_renders_fit_stage_metadata() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");
    let run_id = "5ca1ab1e00000000000000000000000000000000000000000000000000000000";
    let leaf = write_cas_fit_stage(&output, run_id, "parent");

    // Resolve by full path.
    let out = Command::new(&bin)
        .args(["show", &leaf.to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success(), "show fit-stage failed: stderr={}",
        String::from_utf8_lossy(&out.stderr));
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("fit_stage"), "kind label missing: {}", s);
    assert!(s.contains("scout"),     "stage name missing: {}", s);
    assert!(s.contains("if2"),       "method missing: {}", s);
    assert!(s.contains("-123.45"),   "best_loglik missing: {}", s);

    // Resolve by run_id prefix.
    let out = Command::new(&bin)
        .args(["show", "5ca1ab1e", "--root", &output.to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success(), "show by run_id prefix failed: stderr={}",
        String::from_utf8_lossy(&out.stderr));
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("scout"));
}


/// Test (5 from review): `camdl label` works on a new-format sim leaf —
/// the label persists on the leaf's `RunRecord.provenance.label`. The
/// profile case (label → the profile-base sidecar, NOT per-leaf) is covered
/// by `profile_priors::label_command_relabels_profile_sidecar`, where the
/// profile-running harness lives.
#[test]
fn label_works_on_sim_runs() {
    let bin = skip_if_missing_binary();
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
}

/// `camdl reindex` rebuilds `<root>/index.json` from the live run.json files,
/// and the index accelerates a subsequent `show` without changing its output.
/// (gh#147 M4: the derived index.)
#[test]
fn reindex_builds_index_and_show_still_resolves() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");

    // Cache two sims.
    for seed in ["42", "99"] {
        Command::new(&bin)
            .args(["simulate", &golden_sir_basic().to_string_lossy(),
                   "--scenario", "baseline", "--seed", seed, "--cas",
                   "--output-dir", &output.to_string_lossy(),
                   "-o", &tmp.path().join("t.tsv").to_string_lossy()])
            .status().expect("spawn");
    }

    // No index yet (the writer does not emit one in M4 piece 1).
    assert!(!output.join("index.json").exists(), "no index before reindex");

    let out = Command::new(&bin)
        .args(["reindex", &output.to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success(), "reindex should succeed: {}",
        String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("reindexed 2 run"), "summary should count 2 runs: {}", stdout);
    assert!(output.join("index.json").exists(), "index.json written");
    assert!(!output.join("index.json.tmp").exists(),
        "no index.json.tmp left after a successful atomic write");

    // The index entries point at the live leaves: show by short hash resolves.
    let dir = walkdir(&output.join("sims")).into_iter()
        .find(|p| p.join("run.json").exists()).unwrap();
    let short = &read_meta(&dir)["run_id"].as_str().unwrap()[..8].to_string();
    let out = Command::new(&bin)
        .args(["show", short, "--root", &output.to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success(), "show via index should resolve: {}",
        String::from_utf8_lossy(&out.stderr));
}

/// Invariant 1 (miss → walk → repair): a leaf added out of band AFTER an index
/// exists must still be found by `show` — never reported "no match" because the
/// index lacks it.
#[test]
fn out_of_band_leaf_is_found_via_walk_fallback() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");

    // Cache one sim and build an index that knows only about it.
    Command::new(&bin)
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline", "--seed", "1", "--cas",
               "--output-dir", &output.to_string_lossy(),
               "-o", &tmp.path().join("t.tsv").to_string_lossy()])
        .status().expect("spawn");
    let st = Command::new(&bin)
        .args(["reindex", &output.to_string_lossy()])
        .status().expect("spawn");
    assert!(st.success());

    // Add a SECOND sim out of band (seed 2). The index from the reindex above
    // does NOT contain it.
    Command::new(&bin)
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline", "--seed", "2", "--cas",
               "--output-dir", &output.to_string_lossy(),
               "-o", &tmp.path().join("t.tsv").to_string_lossy()])
        .status().expect("spawn");

    // Find the out-of-band leaf's run_id (seed_2 segment).
    let oob_dir = walkdir(&output.join("sims")).into_iter()
        .find(|p| p.join("run.json").exists()
              && p.file_name().unwrap().to_string_lossy().starts_with("seed_2-"))
        .expect("seed_2 leaf");
    let oob_short = &read_meta(&oob_dir)["run_id"].as_str().unwrap()[..8].to_string();

    // `show` of the out-of-band hash must succeed via the walk fallback, NOT
    // report "no match" because the (stale) index lacks the entry.
    let out = Command::new(&bin)
        .args(["show", oob_short, "--root", &output.to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success(),
        "out-of-band leaf must resolve via walk fallback: stderr={}",
        String::from_utf8_lossy(&out.stderr));
    assert!(!String::from_utf8_lossy(&out.stderr).contains("no run matches"),
        "must not report 'no match' for an out-of-band leaf");
}

/// Invariant 2 (stale → drop): a leaf whose entry is in the index but whose
/// directory was removed must NOT resolve to the dead path; `show` reports a
/// clean "no match" instead of pointing at the missing leaf.
#[test]
fn removed_leaf_does_not_resolve_to_dead_path() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");

    Command::new(&bin)
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline", "--seed", "1", "--cas",
               "--output-dir", &output.to_string_lossy(),
               "-o", &tmp.path().join("t.tsv").to_string_lossy()])
        .status().expect("spawn");
    let st = Command::new(&bin)
        .args(["reindex", &output.to_string_lossy()])
        .status().expect("spawn");
    assert!(st.success());

    let dir = walkdir(&output.join("sims")).into_iter()
        .find(|p| p.join("run.json").exists()).unwrap();
    let short = read_meta(&dir)["run_id"].as_str().unwrap()[..8].to_string();
    // It resolves while present.
    assert!(Command::new(&bin)
        .args(["show", &short, "--root", &output.to_string_lossy()])
        .status().expect("spawn").success(), "resolves while present");

    // Remove the leaf out of band; the index entry is now stale.
    std::fs::remove_dir_all(&dir).unwrap();

    let out = Command::new(&bin)
        .args(["show", &short, "--root", &output.to_string_lossy()])
        .output().expect("spawn");
    assert!(!out.status.success(),
        "a removed leaf must not resolve (no dead path)");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("no run matches"),
        "expected clean 'no run matches', got: {}", stderr);
}

// ─── SimEnsemble (multi-cell combined TSV) tests ─────────────────────────────

/// The single ensemble leaf written by a multi-cell run (under
/// `<out>/ensembles/`). Panics if not exactly one.
fn sole_ensemble_leaf(out: &Path) -> PathBuf {
    let leaves: Vec<_> = walkdir(&out.join("ensembles")).into_iter()
        .filter(|p| p.join("run.json").exists()).collect();
    assert_eq!(leaves.len(), 1, "expected exactly one ensemble leaf, got {}", leaves.len());
    leaves.into_iter().next().unwrap()
}

/// Count-in-the-key: a 3-replicate run and a 4-replicate run produce DIFFERENT
/// ensemble `run_id`s — the combined TSV has a different number of rows, so the
/// cell set (and its count) is in the key (the n_trajectories collision class).
#[test]
fn ensemble_cell_count_is_in_the_run_id() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();

    let out3 = tmp.path().join("out3");
    let st = Command::new(&bin)
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline", "--seed", "42", "--replicates", "3",
               "--output-dir", &out3.to_string_lossy()])
        .status().expect("spawn");
    assert!(st.success(), "3-replicate simulate should succeed");

    let out4 = tmp.path().join("out4");
    let st = Command::new(&bin)
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline", "--seed", "42", "--replicates", "4",
               "--output-dir", &out4.to_string_lossy()])
        .status().expect("spawn");
    assert!(st.success(), "4-replicate simulate should succeed");

    let id3 = read_meta(&sole_ensemble_leaf(&out3))["run_id"].as_str().unwrap().to_string();
    let id4 = read_meta(&sole_ensemble_leaf(&out4))["run_id"].as_str().unwrap().to_string();
    assert_ne!(id3, id4,
        "3 vs 4 replicates must give DIFFERENT ensemble run_ids (cell count in the key)");

    // Sanity: the leaf-side count-in-the-key still holds (3 vs 4 Sim leaves).
    let n3 = walkdir(&out3.join("sims")).into_iter()
        .filter(|p| p.join("run.json").exists()).count();
    let n4 = walkdir(&out4.join("sims")).into_iter()
        .filter(|p| p.join("run.json").exists()).count();
    assert_eq!((n3, n4), (3, 4), "per-cell Sim leaf counts must be 3 and 4");
}

/// Round-trip: a 3-replicate run writes ONE ensemble leaf whose `deps` are
/// exactly the 3 per-cell `Sim` leaf run_ids; `cat <ensemble>` emits the
/// combined wide-format TSV (with a `replicate` column, 3 replicates),
/// byte-identical to the `-o` mirror; `list --kind ensemble` surfaces it.
#[test]
fn ensemble_round_trip_deps_cat_and_list() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("out");
    let mirror = tmp.path().join("combined.tsv");

    let st = Command::new(&bin)
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline", "--seed", "42", "--replicates", "3",
               "--output-dir", &out.to_string_lossy(),
               "-o", &mirror.to_string_lossy()])
        .status().expect("spawn");
    assert!(st.success(), "3-replicate simulate should succeed");

    // The 3 per-cell Sim leaf run_ids.
    let sim_leaves: Vec<_> = walkdir(&out.join("sims")).into_iter()
        .filter(|p| p.join("run.json").exists()).collect();
    assert_eq!(sim_leaves.len(), 3, "expected 3 Sim leaves");
    let mut sim_ids: Vec<String> = sim_leaves.iter()
        .map(|d| read_meta(d)["run_id"].as_str().unwrap().to_string()).collect();
    sim_ids.sort();

    // The ensemble leaf: kind = sim_ensemble, deps = the 3 Sim leaves.
    let ens = sole_ensemble_leaf(&out);
    let ens_meta = read_meta(&ens);
    assert_eq!(ens_meta["kind"].as_str().unwrap(), "sim_ensemble");
    let deps = ens_meta["deps"].as_array().expect("ensemble has deps");
    assert_eq!(deps.len(), 3, "ensemble deps must be the 3 per-cell Sim leaves");
    let mut dep_ids: Vec<String> = deps.iter()
        .map(|d| d["run_id"].as_str().unwrap().to_string()).collect();
    dep_ids.sort();
    assert_eq!(dep_ids, sim_ids, "ensemble deps must be exactly the 3 Sim leaf run_ids");
    for d in deps {
        assert_eq!(d["kind"].as_str().unwrap(), "sim");
        assert_eq!(d["artifact"].as_str().unwrap(), "traj.tsv");
    }

    // `cat <ensemble-hash>` emits the combined TSV byte-identical to the `-o`
    // mirror, and that TSV has a `replicate` column with 3 replicates.
    let ens_id = ens_meta["run_id"].as_str().unwrap();
    let cat = Command::new(&bin)
        .args(["cat", &ens_id[..16], "--root", &out.to_string_lossy()])
        .output().expect("spawn");
    assert!(cat.status.success(), "cat <ensemble> failed: {}",
        String::from_utf8_lossy(&cat.stderr));
    let mirror_bytes = std::fs::read(&mirror).unwrap();
    assert!(!mirror_bytes.is_empty(), "the -o mirror must be non-empty");
    assert_eq!(cat.stdout, mirror_bytes,
        "cat <ensemble> must equal the -o combined TSV byte-for-byte");

    // The combined TSV header carries a `replicate` column, and the data has
    // exactly 3 distinct replicate values.
    let text = String::from_utf8(mirror_bytes).unwrap();
    let header = text.lines().find(|l| !l.starts_with('#')).expect("a header line");
    assert_eq!(header.split('\t').next().unwrap(), "replicate",
        "multi-replicate combined TSV must lead with a `replicate` column, got: {header}");
    let reps: std::collections::BTreeSet<&str> = text.lines()
        .filter(|l| !l.starts_with('#'))
        .skip(1) // header
        .filter_map(|l| l.split('\t').next())
        .filter(|c| !c.is_empty())
        .collect();
    assert_eq!(reps, ["1", "2", "3"].into_iter().collect(),
        "combined TSV must contain replicates 1,2,3, got {reps:?}");

    // `list --kind ensemble` surfaces the ensemble (table to stderr; json to
    // stdout, one record per line).
    let list = Command::new(&bin)
        .args(["list", &out.to_string_lossy(), "--kind", "ensemble", "--format", "json"])
        .output().expect("spawn");
    assert!(list.status.success(), "list --kind ensemble failed: {}",
        String::from_utf8_lossy(&list.stderr));
    // The json mode emits one record per line; collect the `sim_ensemble`
    // records (other kinds' printers may emit an empty `[]` line — tolerate it).
    let list_out = String::from_utf8(list.stdout).unwrap();
    let ensemble_rows: Vec<serde_json::Value> = list_out.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter(|v| v.get("kind").and_then(|k| k.as_str()) == Some("sim_ensemble"))
        .collect();
    assert_eq!(ensemble_rows.len(), 1,
        "list --kind ensemble must surface exactly the one ensemble, got: {list_out}");
    assert_eq!(ensemble_rows[0]["run_id"].as_str().unwrap(), ens_id);
}

/// Single-run simulate writes NO ensemble (the one Sim leaf is the whole
/// thing).
#[test]
fn single_run_writes_no_ensemble() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("out");
    let st = Command::new(&bin)
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline", "--seed", "42",
               "--output-dir", &out.to_string_lossy(),
               "-o", &tmp.path().join("c.tsv").to_string_lossy()])
        .status().expect("spawn");
    assert!(st.success(), "single-run simulate should succeed");
    let ensembles: Vec<_> = walkdir(&out.join("ensembles")).into_iter()
        .filter(|p| p.join("run.json").exists()).collect();
    assert!(ensembles.is_empty(), "a single-run simulate must write NO ensemble");
}

/// Item C: `simulate` (single AND multi-cell) with NO `-o` writes NOTHING to
/// stdout — the CAS leaf/ensemble are the system of record. The `-o` mirror
/// still works, and the leaf/ensemble exist either way.
#[test]
fn simulate_writes_nothing_to_stdout_without_output_flag() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();

    // Single-run, no -o: empty stdout, one Sim leaf, no ensemble.
    let out1 = tmp.path().join("out1");
    let single = Command::new(&bin)
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline", "--seed", "42",
               "--output-dir", &out1.to_string_lossy()])
        .output().expect("spawn");
    assert!(single.status.success(), "single-run simulate (no -o) should succeed: {}",
        String::from_utf8_lossy(&single.stderr));
    assert!(single.stdout.is_empty(),
        "single-run simulate with no -o must write NOTHING to stdout, got {} bytes",
        single.stdout.len());
    assert_eq!(walkdir(&out1.join("sims")).into_iter()
        .filter(|p| p.join("run.json").exists()).count(), 1,
        "the Sim leaf must still be written");

    // Multi-cell, no -o: empty stdout, but the leaves AND ensemble exist.
    let out3 = tmp.path().join("out3");
    let multi = Command::new(&bin)
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline", "--seed", "42", "--replicates", "3",
               "--output-dir", &out3.to_string_lossy()])
        .output().expect("spawn");
    assert!(multi.status.success(), "multi-cell simulate (no -o) should succeed: {}",
        String::from_utf8_lossy(&multi.stderr));
    assert!(multi.stdout.is_empty(),
        "multi-cell simulate with no -o must write NOTHING to stdout, got {} bytes",
        multi.stdout.len());
    assert_eq!(walkdir(&out3.join("sims")).into_iter()
        .filter(|p| p.join("run.json").exists()).count(), 3,
        "the 3 Sim leaves must still be written");
    let _ = sole_ensemble_leaf(&out3); // ensemble exists even with no -o

    // With -o, the mirror is written and matches `cat <ensemble>`.
    let out3b = tmp.path().join("out3b");
    let mirror = tmp.path().join("m.tsv");
    let mirrored = Command::new(&bin)
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline", "--seed", "42", "--replicates", "3",
               "--output-dir", &out3b.to_string_lossy(),
               "-o", &mirror.to_string_lossy()])
        .output().expect("spawn");
    assert!(mirrored.status.success(), "multi-cell simulate (-o) should succeed");
    assert!(mirrored.stdout.is_empty(),
        "even WITH -o, simulate writes the trajectory to the file, not stdout");
    let mirror_bytes = std::fs::read(&mirror).unwrap();
    let ens = sole_ensemble_leaf(&out3b);
    let ens_tsv = std::fs::read(ens.join("ensemble.tsv")).unwrap();
    assert_eq!(mirror_bytes, ens_tsv,
        "the -o mirror must be byte-identical to the stored ensemble.tsv");
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Collect all directory paths under `root` (non-recursive children of
/// each level; bounded depth is fine for our 3-level CAS layout).
/// Every directory at any depth under `root`. Callers filter to leaves with
/// `.filter(|p| p.join("run.json").exists())`. Depth-agnostic so it handles
/// the factored 5-level CAS path (model/config/params/scenario/seed), not
/// just the legacy 3-level layout.
/// The sole leaf (a dir with a `run.json`) under `sims`, panicking if not
/// exactly one.
fn sole_leaf(sims: &Path) -> PathBuf {
    let dirs: Vec<_> = walkdir(sims).into_iter()
        .filter(|p| p.join("run.json").exists()).collect();
    assert_eq!(dirs.len(), 1, "expected exactly one leaf under {:?}, got {}", sims, dirs.len());
    dirs.into_iter().next().unwrap()
}

/// A model with a `#[lineage]`-annotated transition (required by
/// `--event-log`), compiled to IR. Returns `None` if camdlc is unavailable
/// (the test then skips, matching `lineage_e2e`).
fn compile_lineage_model(bin: &Path, tmp: &Path) -> Option<PathBuf> {
    const SIR_LINEAGE: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.001, 5.0]
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 10000]
}
let N = S + I + R
transitions {
  #[lineage]
  infection : S --> I @ beta * S * I / N
  recovery  : I --> R @ gamma * I
}
init { S = 499  I = 1 }
simulate { from = 0 'days  to = 60 'days }
"#;
    let model = tmp.join("sir_lineage.camdl");
    std::fs::write(&model, SIR_LINEAGE).unwrap();
    let ir = tmp.join("sir_lineage.ir.json");
    let compiled = Command::new(bin)
        .args(["compile", model.to_str().unwrap(), "-o", ir.to_str().unwrap()])
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().expect("spawn compile");
    if !compiled.status.success() {
        eprintln!("skipping: camdl compile failed (camdlc unavailable): {}",
            String::from_utf8_lossy(&compiled.stderr));
        return None;
    }
    Some(ir)
}

/// `simulate --event-log` records the event log as a content-addressed
/// artifact that sits ALONGSIDE `traj.tsv` in the SAME `Sim` leaf — not as a
/// loose file. The recorder is passive (Tier 2a), so the leaf's run_id and
/// `traj.tsv` bytes are identical to a plain `simulate` at the same seed; the
/// event log is just an additional declared artifact. `--event-log PATH` also
/// writes PATH as a byte-identical mirror (symmetric with `-o` for the
/// trajectory).
#[test]
fn event_log_lands_in_sim_leaf_alongside_traj() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let Some(ir) = compile_lineage_model(&bin, tmp.path()) else { return; };
    let ir_s = ir.to_string_lossy().into_owned();
    let common = ["--backend", "gillespie", "--seed", "7",
                  "--param", "beta=0.6", "--param", "gamma=0.2", "--param", "N0=500"];

    // (1) plain simulate → leaf with traj.tsv only.
    let out_plain = tmp.path().join("plain");
    let st = Command::new(&bin)
        .args(["simulate", &ir_s]).args(common)
        .args(["--output-dir", &out_plain.to_string_lossy(),
               "-o", &tmp.path().join("traj_plain.tsv").to_string_lossy()])
        .status().expect("spawn");
    assert!(st.success(), "plain simulate should succeed");
    let plain_leaf = sole_leaf(&out_plain.join("sims"));
    let plain_run_id = read_meta(&plain_leaf)["run_id"].as_str().unwrap().to_string();
    let plain_traj = std::fs::read(plain_leaf.join("traj.tsv")).unwrap();
    assert!(!plain_leaf.join("event_log.tsv").exists(),
        "plain simulate must NOT write event_log.tsv");

    // (2) simulate --event-log PATH → leaf with traj.tsv + event_log.tsv, same
    //     run_id and identical traj bytes; PATH is a byte-identical mirror.
    let out_el = tmp.path().join("el");
    let mirror = tmp.path().join("mirror.tsv");
    let st = Command::new(&bin)
        .args(["simulate", &ir_s]).args(common)
        .args(["--output-dir", &out_el.to_string_lossy(),
               "--event-log", &mirror.to_string_lossy()])
        .status().expect("spawn");
    assert!(st.success(), "simulate --event-log should succeed");
    let el_leaf = sole_leaf(&out_el.join("sims"));
    let el_meta = read_meta(&el_leaf);

    assert_eq!(el_meta["run_id"].as_str().unwrap(), plain_run_id,
        "--event-log must not change the sim run_id (passive recorder, Tier 2a)");
    assert_eq!(std::fs::read(el_leaf.join("traj.tsv")).unwrap(), plain_traj,
        "trajectory must be byte-identical with/without --event-log");
    assert!(el_leaf.join("event_log.tsv").exists(),
        "event_log.tsv should sit alongside traj.tsv in the leaf");
    assert!(el_meta["artifacts"].as_object().unwrap().contains_key("event_log.tsv"),
        "event_log.tsv must be declared in the run.json artifacts exact-set");
    assert_eq!(std::fs::read(&mirror).unwrap(),
               std::fs::read(el_leaf.join("event_log.tsv")).unwrap(),
        "--event-log PATH mirror must be byte-identical to the leaf's event_log.tsv");
}

/// A `--label` passed to `simulate` must end up on the leaf's
/// `provenance.label` EVEN WHEN the trajectory is a CAS cache hit — the
/// content-addressed dedup must not silently drop the user's explicit label.
/// (Matches `fit run`, which keeps its label current on an all-cache-hit
/// rerun.) Conversely, a label-less rerun must NOT wipe an existing label.
#[test]
fn simulate_label_applies_on_cached_leaf() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("store");
    let gold = golden_sir_basic();
    let run = |label: Option<&str>| {
        let mut args = vec!["simulate".to_string(), gold.to_string_lossy().into_owned(),
            "--scenario".into(), "baseline".into(), "--seed".into(), "123".into(),
            "--output-dir".into(), out.to_string_lossy().into_owned()];
        if let Some(l) = label { args.push("--label".into()); args.push(l.into()); }
        assert!(Command::new(&bin).args(&args).status().expect("spawn").success());
    };

    // 1. fresh run, no label → null label.
    run(None);
    let leaf = sole_leaf(&out.join("sims"));
    assert!(read_meta(&leaf)["provenance"]["label"].is_null(),
        "no --label → null label");

    // 2. re-run SAME params WITH --label: the cached leaf must adopt it.
    run(Some("added later"));
    assert_eq!(read_meta(&leaf)["provenance"]["label"].as_str(), Some("added later"),
        "simulate --label on a cached leaf must apply the label (dedup must not drop it)");

    // 3. label-less rerun must PRESERVE the existing label (absence ≠ clear).
    run(None);
    assert_eq!(read_meta(&leaf)["provenance"]["label"].as_str(), Some("added later"),
        "a label-less rerun must not wipe an existing label");
}

/// `camdl list` collapses ensemble members: a multi-replicate `simulate`
/// writes N per-cell `Sim` leaves + one `SimEnsemble`, and the default/`all`
/// view shows ONLY the ensemble (not one row per replicate). `--kind sim` still
/// surfaces the individual leaves. `--root DIR` is accepted as an alias for the
/// positional ROOT (consistency with `cat`/`show`).
#[test]
fn list_collapses_ensemble_members_and_accepts_root_flag() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let store = tmp.path().join("store");
    let store_s = store.to_string_lossy().into_owned();
    assert!(Command::new(&bin)
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline", "--seed", "5", "--replicates", "3",
               "--output-dir", &store_s])
        .status().expect("spawn").success());

    // JSONL: one `RunRecord` per line on stdout. Parse each line and inspect
    // the TOP-LEVEL `kind` — a substring match would be fooled by the
    // ensemble record's embedded `deps`, each of which carries `"kind":"sim"`.
    let count_kind = |args: &[&str], kind: &str| -> usize {
        let out = Command::new(&bin).args(args).output().expect("spawn");
        String::from_utf8_lossy(&out.stdout).lines()
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .filter(|v| v["kind"] == kind)
            .count()
    };

    // Default view (positional ROOT): one ensemble, zero loose sim rows.
    assert_eq!(count_kind(&["list", &store_s, "--format", "json"], "sim_ensemble"), 1,
        "default list must show the ensemble");
    assert_eq!(count_kind(&["list", &store_s, "--format", "json"], "sim"), 0,
        "default list must NOT print one row per replicate (members collapse into the ensemble)");

    // `--root DIR` alias behaves identically to the positional.
    assert_eq!(count_kind(&["list", "--root", &store_s, "--format", "json"], "sim_ensemble"), 1,
        "`list --root DIR` must work like the positional ROOT");

    // `--kind sim`: the individual per-cell leaves ARE surfaced (3 replicates).
    assert_eq!(count_kind(&["list", "--root", &store_s, "--kind", "sim", "--format", "json"], "sim"), 3,
        "`list --kind sim` must surface all 3 replicate leaves");
}

/// The store banner printed after a `simulate` includes the `--output-dir`
/// prefix (a copy-paste-ready path), not just the store-relative `sims/…` tail,
/// plus the `camdl cat <run_id>` that reads the run back.
#[test]
fn stored_banner_includes_output_dir_prefix_and_cat_hint() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let store = tmp.path().join("store");
    let out = Command::new(&bin)
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline", "--seed", "7",
               "--output-dir", &store.to_string_lossy()])
        .output().expect("spawn");
    let stderr = String::from_utf8_lossy(&out.stderr);
    let prefix = format!("{}/sims", store.to_string_lossy());
    assert!(stderr.contains(&prefix),
        "the store banner must include the --output-dir prefix '{prefix}':\n{stderr}");
    assert!(stderr.contains("camdl cat "),
        "the banner must tell the user how to read the run back:\n{stderr}");
}

/// `--stdout` streams the trajectory TSV to stdout and writes NO store leaf
/// and NO banner — the escape hatch for piping.
#[test]
fn stdout_streams_trajectory_and_skips_the_store() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let store = tmp.path().join("store");
    let out = Command::new(&bin)
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline", "--seed", "7",
               "--output-dir", &store.to_string_lossy(), "--stdout"])
        .output().expect("spawn");
    assert!(out.status.success(), "simulate --stdout should succeed");
    let stdout = String::from_utf8_lossy(&out.stdout);
    // A wide-format trajectory TSV on stdout: a `# camdl <version>` comment,
    // then the `t\tS\t…` column header, then rows.
    assert!(stdout.lines().any(|l| l.starts_with("t\t")),
        "stdout should carry the trajectory TSV header (t<TAB>…), got:\n{stdout}");
    assert!(stdout.lines().count() > 3, "stdout should carry trajectory rows");
    // No store leaf was written, and no banner pointed at one.
    assert!(!store.join("sims").exists(),
        "--stdout must NOT write a store leaf");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("stored"),
        "--stdout must not print the store banner:\n{stderr}");
}

/// Progress rendering is passive: a multi-cell `simulate` produces a
/// byte-identical trajectory whether progress is `none` or `pretty`. The bars
/// must never perturb the RNG / draw order (the engine's determinism contract).
#[test]
fn progress_mode_does_not_change_trajectories() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let run = |mode: &str, tag: &str| -> Vec<u8> {
        let o = tmp.path().join(format!("traj_{tag}.tsv"));
        let st = Command::new(&bin)
            .args(["simulate", &golden_sir_basic().to_string_lossy(),
                   "--scenario", "baseline", "--seed", "7", "--replicates", "3",
                   "--output-dir", &tmp.path().join(tag).to_string_lossy(),
                   "-o", &o.to_string_lossy(), "--progress", mode])
            .status().expect("spawn");
        assert!(st.success(), "simulate --progress {mode} should succeed");
        std::fs::read(&o).unwrap()
    };
    assert_eq!(run("none", "n"), run("pretty", "p"),
        "progress mode must not change the trajectory (bars must be RNG-passive)");
}

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
