//! End-to-end tests for `camdl simulate`'s store and `camdl list/show/cat`.
//!
//! These shell out to the built `camdl` binary in `target/release/`
//! and exercise the full pipeline: real hash computation, real cache
//! lookups, real directory writes.

use std::path::{Path, PathBuf};
use std::process::Command;

mod sim_store_cells;
use sim_store_cells::SimCell;

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
/// suitable for `--scenario baseline --seed N`.
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

// ── Run-store helpers ──────────────────────────────────────────────────────

/// The stored simulation cells under the run store `store`, leaving out any
/// cell a repair moved to `.quarantine/`.
fn live_cells(store: &Path) -> Vec<SimCell> {
    sim_store_cells::sim_cells(store).into_iter()
        .filter(|c| !c.dir.components().any(|p| p.as_os_str() == ".quarantine"))
        .collect()
}

/// The one live simulation cell under `store`; panics unless exactly one.
fn sole_cell(store: &Path) -> SimCell {
    let cells = live_cells(store);
    assert_eq!(cells.len(), 1, "expected exactly one cell under {}, got {:?}",
        store.display(), cells.iter().map(|c| &c.dir).collect::<Vec<_>>());
    cells.into_iter().next().unwrap()
}

/// The `sim/` folder of the run store `store`: `<store>/<build>/sim`, the one
/// a single build writes.
fn sim_dir(store: &Path) -> PathBuf {
    let builds: Vec<PathBuf> = std::fs::read_dir(store).unwrap().flatten()
        .map(|e| e.path())
        .filter(|p| p.join("sim").is_dir())
        .collect();
    assert_eq!(builds.len(), 1, "one build folder with a sim/ under {}: {builds:?}", store.display());
    builds[0].join("sim")
}

#[test]
fn cas_first_run_writes_cache_and_metadata() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");

    let status = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline",
               "--seed", "42",
               "--store", &output.to_string_lossy(),
               "-o", &tmp.path().join("traj.tsv").to_string_lossy()])
        .status()
        .expect("spawn");
    assert!(status.success(), "first run should succeed");

    // Exactly one cell in the store.
    let cell = sole_cell(&output);
    assert!(cell.dir.join("traj.tsv").exists(), "traj.tsv should be written");
    assert!(cell.dir.join("run.json").exists(), "run.json should be written");

    // identity.json names the cell; run.json records the run.
    assert_eq!(cell.identity["kind"], "sim", "identity.json: kind=sim");
    assert_eq!(cell.run_id.len(), 64);
    let meta = cell.run_json();
    assert_eq!(meta["status"], "completed");
    assert_eq!(cell.seed(), 42);
    assert_eq!(cell.scenario_dir_name(), "baseline");
    // The invocation that wrote the cell keeps its argv.
    let inv = meta["invocation"].as_u64().expect("run.json names its invocation");
    let inv: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        output.join(".invocations").join(format!("{inv}.json"))).unwrap()).unwrap();
    assert!(inv["argv"].as_array().unwrap().len() >= 4);
    // `ambient.build` is the engine's build fingerprint (ENGINE_ID, 128 bits
    // of hex).
    let engine = cell.identity["ambient"]["build"].as_str().unwrap();
    assert!(engine.len() == 32 && engine.bytes().all(|b| b.is_ascii_hexdigit()),
        "ambient.build is the build fingerprint: {engine}");
    assert!(meta["started"].as_str().is_some() && meta["finished"].as_str().is_some());
}

#[test]
fn cas_second_identical_run_does_not_rewrite_leaf() {
    // A second, byte-identical run finds the stored cell and does not
    // re-write its `traj.tsv`. The proof is the unchanged mtime.
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");

    let run_once = || {
        Command::new(&bin).env_remove("CLICOLOR_FORCE")
            .args(["simulate", &golden_sir_basic().to_string_lossy(),
                   "--scenario", "baseline",
                   "--seed", "42",
                   "--store", &output.to_string_lossy(),
                   "-o", &tmp.path().join("traj.tsv").to_string_lossy()])
            .output()
            .expect("spawn")
    };

    let first = run_once();
    assert!(first.status.success());
    let stderr1 = String::from_utf8_lossy(&first.stderr);
    assert!(stderr1.contains("stored"), "first run stderr should report the run was stored: {}", stderr1);

    // Wait long enough that the filesystem mtime would differ if rewritten.
    let cache_path = sole_cell(&output).dir.join("traj.tsv");
    let mtime1 = std::fs::metadata(&cache_path).unwrap().modified().unwrap();

    std::thread::sleep(std::time::Duration::from_millis(10));

    let second = run_once();
    assert!(second.status.success());

    // mtime unchanged — the second run must not re-write the cell.
    let mtime2 = std::fs::metadata(&cache_path).unwrap().modified().unwrap();
    assert_eq!(mtime1, mtime2, "a stored cell must not be overwritten by an identical run");
}

#[test]
fn cas_different_seed_new_cache_entry() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");

    for seed in ["42", "43"] {
        let st = Command::new(&bin).env_remove("CLICOLOR_FORCE")
            .args(["simulate", &golden_sir_basic().to_string_lossy(),
                   "--scenario", "baseline",
                   "--seed", seed,
                   "--store", &output.to_string_lossy(),
                   "-o", &tmp.path().join("traj.tsv").to_string_lossy()])
            .status().expect("spawn");
        assert!(st.success());
    }

    let cells = live_cells(&output);
    assert_eq!(cells.len(), 2, "should have two separate seed cells");

    // Each cell's identity records its seed.
    let mut seeds: Vec<u64> = cells.iter().map(|c| c.seed()).collect();
    seeds.sort();
    assert_eq!(seeds, [42, 43], "got {seeds:?}");
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
        let st = Command::new(&bin).env_remove("CLICOLOR_FORCE")
            .args(["simulate", &model.to_string_lossy(),
                   "--scenario", "baseline",
                   "--seed", "1",
                   "--store", &output.to_string_lossy(),
                   "-o", &tmp.path().join("traj.tsv").to_string_lossy()])
            .status().expect("spawn");
        assert!(st.success(), "simulate should succeed");
    };
    run(&path_a);
    run(&path_b);

    // Two distinct cells — not one collision.
    let cells = live_cells(&output);
    assert_eq!(cells.len(), 2,
        "two structurally different models must produce two cells, \
         not collide to one (gh#135)");

    // The compiled model's digest (structural whole-IR digest) must be
    // recorded in each cell's identity, 64-hex, and distinct between the two
    // structurally different models.
    let mut model_hashes: Vec<String> = cells.iter().map(|c| {
        c.identity["identity"]["model"]["digest"].as_str().unwrap().to_string()
    }).collect();
    model_hashes.sort();
    for mh in &model_hashes {
        assert_eq!(mh.len(), 64, "model digest is 64-hex");
    }
    assert_ne!(model_hashes[0], model_hashes[1],
        "two structurally different models must record distinct model \
         digests (gh#135/gh#147)");

    // And the trajectories themselves must differ — the symptom users saw.
    let trajs: Vec<Vec<u8>> = cells.iter()
        .map(|c| std::fs::read(c.dir.join("traj.tsv")).unwrap()).collect();
    assert_ne!(trajs[0], trajs[1],
        "doubling the recovery rate must change the trajectory; identical \
         bytes means one model was served the other's cached result (gh#135)");
}

#[test]
fn cas_multi_seeds_writes_one_leaf_per_seed() {
    // Storing is the default and a
    // multi-cell run writes one leaf per cell. `--seeds 1:3` is
    // three seed-slots → three leaves with
    // three distinct seed-level segments.
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");

    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline",
               "--seeds", "1:3",
               "--store", &output.to_string_lossy()])
        .output()
        .expect("spawn");
    assert!(out.status.success(), "multi-seed simulate should succeed: {}",
        String::from_utf8_lossy(&out.stderr));

    let cells = live_cells(&output);
    assert_eq!(cells.len(), 3, "three seeds must produce three cells");

    // Three distinct run_ids (the seed is in the key — count-in-the-key).
    let mut run_ids: Vec<String> = cells.iter().map(|c| c.run_id.clone()).collect();
    run_ids.sort();
    run_ids.dedup();
    assert_eq!(run_ids.len(), 3, "three seeds must yield three distinct run_ids");

    // The seeds are 1, 2, 3 (explicit --seeds are used verbatim).
    let mut seeds: Vec<u64> = cells.iter().map(|c| c.seed()).collect();
    seeds.sort();
    assert_eq!(seeds, [1, 2, 3], "got {seeds:?}");
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
    let st = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline", "--seed", "42", "--replicates", "3",
               "--store", &out3.to_string_lossy(),
               "-o", &tmp.path().join("c3.tsv").to_string_lossy()])
        .status().expect("spawn");
    assert!(st.success(), "3-replicate simulate should succeed");
    let cells3 = live_cells(&out3);
    assert_eq!(cells3.len(), 3, "3 replicates must write 3 cells");
    let mut ids3: Vec<String> = cells3.iter().map(|c| c.run_id.clone()).collect();
    ids3.sort(); ids3.dedup();
    assert_eq!(ids3.len(), 3, "3 replicates must have 3 distinct run_ids (seed in key)");

    // 1 run (no --replicates) → exactly one leaf.
    let out1 = tmp.path().join("out1");
    let st = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline", "--seed", "42",
               "--store", &out1.to_string_lossy(),
               "-o", &tmp.path().join("c1.tsv").to_string_lossy()])
        .status().expect("spawn");
    assert!(st.success(), "single-run simulate should succeed");
    assert_eq!(live_cells(&out1).len(), 1, "a lone simulate run must write exactly one cell");
}

/// The load-bearing invariant: the multi-cell path (`simulate --seeds …`,
/// whose sink wraps the shared `CasSink`) and the single-cell path
/// (`simulate --seed N`) write leaves at the SAME store paths with the SAME
/// `run_id`s and BYTE-IDENTICAL `traj.tsv` for the same (model, config,
/// params, scenario, process_seed) cells. Both resolve identity through one
/// `CasSink`, so divergence here means the two paths resolved identity
/// differently — a silent-wrong-answer class bug (a lone run's leaf would
/// never be a hit for the same cell of an ensemble, or vice versa).
#[test]
fn multi_cell_and_single_cell_leaves_are_byte_identical() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();

    let params = tmp.path().join("params.toml");
    std::fs::write(&params, "beta = 0.3\ngamma = 0.1\nN0 = 1000\nI0 = 10\n").unwrap();

    let multi_out = tmp.path().join("multi_out");
    let single_out = tmp.path().join("single_out");

    // One multi-cell run over an explicit seed list.
    let st = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline",
               "--params", &params.to_string_lossy(),
               "--seeds", "10,11,12",
               "--store", &multi_out.to_string_lossy(),
               "-o", &tmp.path().join("c.tsv").to_string_lossy()])
        .status().expect("spawn");
    assert!(st.success(), "simulate --seeds should succeed");

    // The same three cells, one lone run each, into a second store.
    for seed in ["10", "11", "12"] {
        let st = Command::new(&bin).env_remove("CLICOLOR_FORCE")
            .args(["simulate", &golden_sir_basic().to_string_lossy(),
                   "--scenario", "baseline",
                   "--params", &params.to_string_lossy(),
                   "--seed", seed,
                   "--store", &single_out.to_string_lossy()])
            .status().expect("spawn");
        assert!(st.success(), "simulate --seed {seed} should succeed");
    }

    // Enumerate the multi-cell cells; for each, the same store-relative path
    // must exist in the single-run store with the same run_id and identical
    // traj bytes.
    let multi_cells = live_cells(&multi_out);
    assert_eq!(multi_cells.len(), 3, "expected 3 multi-cell cells");

    for mc in &multi_cells {
        let rel = mc.dir.strip_prefix(&multi_out).unwrap();
        let sl = single_out.join(rel);
        assert!(sl.join("identity.json").exists(),
            "single-run store missing the same-path cell: {}", rel.display());

        let sid = sim_store_cells::read_json(&sl.join("identity.json"))["run_id"]
            .as_str().unwrap().to_string();
        assert_eq!(mc.run_id, sid, "run_id mismatch at {}", rel.display());

        let mbytes = std::fs::read(mc.dir.join("traj.tsv")).unwrap();
        let sbytes = std::fs::read(sl.join("traj.tsv")).unwrap();
        assert_eq!(mbytes, sbytes,
            "traj.tsv bytes differ between the multi-cell and single-cell paths at {}",
            rel.display());
    }
}

/// The model directories (`sim/<name>-<hash>`) directly under `sims`.
fn model_dirs(sims: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(sims).unwrap().flatten()
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    v
}

/// gh#583 item E. Every run of the same cell of a `.camdl` SOURCE model must
/// land on the same store path. The model directory is named for the source
/// stem, never for the compiled IR the source resolved to — the IR cache's
/// content hash, or `camdl_<pid>` with the cache off — which would put the
/// same run_id under a different directory per invocation and miss the
/// directory-scoped cache lookup on every rerun.
#[test]
fn a_camdl_source_lands_on_one_model_directory_with_the_ir_cache_on_or_off() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("sir_basic.camdl");
    std::fs::copy(golden_sir_basic().with_extension("").with_extension("camdl"), &model)
        .unwrap();
    let params = tmp.path().join("params.toml");
    std::fs::write(&params, "beta = 0.3\ngamma = 0.1\nN0 = 1000\nI0 = 10\n").unwrap();
    let store = tmp.path().join("store");

    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["simulate", &model.to_string_lossy(),
               "--scenario", "baseline",
               "--params", &params.to_string_lossy(),
               "--seed", "10",
               "--store", &store.to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success(), "simulate: {}", String::from_utf8_lossy(&out.stderr));
    let sims = sim_dir(&store);
    let after_first = model_dirs(&sims);
    assert_eq!(after_first.len(), 1, "one model directory: {after_first:?}");
    assert!(after_first[0].starts_with("sir_basic-"),
        "named for the model: {after_first:?}");

    // A multi-cell run of the same source, with the IR cache on and then off:
    // the second compiles to a temp file named after the process id, so a
    // directory named for the IR would differ from every other invocation.
    for no_cache in [false, true] {
        let mut cmd = Command::new(&bin);
        cmd.env_remove("CLICOLOR_FORCE");
        cmd.args(["simulate", &model.to_string_lossy(),
                  "--scenario", "baseline",
                  "--params", &params.to_string_lossy(),
                  "--seeds", "10,11",
                  "--store", &store.to_string_lossy()]);
        if no_cache { cmd.env("CAMDL_NO_IR_CACHE", "1"); }
        let out = cmd.output().expect("spawn");
        assert!(out.status.success(), "simulate --seeds: {}", String::from_utf8_lossy(&out.stderr));
        assert_eq!(model_dirs(&sims), after_first,
            "a run with the IR cache {} must land in the first run's model directory",
            if no_cache { "off" } else { "on" });
    }
}

/// gh#592, on the surface that replaced `batch status`: the store view of a
/// multi-cell `--dry-run` over a `.camdl` source with `--output-every`. It
/// must predict the identity the run resolves — the model compiled from
/// source, with the output cadence lowered into it — so a completed grid
/// reports every cell a hit, and one more seed reports exactly one miss, the
/// new seed's cell.
#[test]
fn a_camdl_source_dry_run_counts_the_leaves_the_run_wrote() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("sir_basic.camdl");
    std::fs::copy(golden_sir_basic().with_extension("").with_extension("camdl"), &model)
        .unwrap();
    let params = tmp.path().join("params.toml");
    std::fs::write(&params, "beta = 0.3\ngamma = 0.1\nN0 = 1000\nI0 = 10\n").unwrap();
    let store = tmp.path().join("store");
    let simulate = |seeds: &str, dry_run: bool| -> String {
        let mut cmd = Command::new(&bin);
        cmd.env_remove("CLICOLOR_FORCE");
        cmd.args(["simulate", &model.to_string_lossy(),
                  "--scenario", "baseline",
                  "--params", &params.to_string_lossy(),
                  "--seeds", seeds,
                  "--output-every", "2",
                  "--store", &store.to_string_lossy()]);
        if dry_run { cmd.arg("--dry-run"); }
        let out = cmd.output().expect("spawn");
        assert!(out.status.success(), "simulate --seeds {seeds} (dry run: {dry_run}): {}",
            String::from_utf8_lossy(&out.stderr));
        String::from_utf8_lossy(&out.stderr).into_owned()
    };

    simulate("1,2", false);
    let s = simulate("1,2", true);
    assert!(s.contains("2 cache hits → read back, 0 misses"),
        "a completed .camdl grid reports every cell a hit:\n{s}");

    // One more seed: that cell is the one miss, printed under the store once.
    let s = simulate("1,2,3", true);
    assert!(s.contains("2 cache hits → read back, 1 misses → would simulate"), "{s}");
    let misses: Vec<&str> = s.lines().filter(|l| l.contains("[miss]")).collect();
    assert_eq!(misses.len(), 1, "one miss line: {s}");
    assert!(misses[0].trim_end().ends_with("seed_3"), "the miss is the new seed: {}", misses[0]);
}

#[test]
fn list_shows_cached_runs() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");

    // Cache two runs
    for seed in ["42", "99"] {
        Command::new(&bin).env_remove("CLICOLOR_FORCE")
            .args(["simulate", &golden_sir_basic().to_string_lossy(),
                   "--scenario", "baseline",
                   "--seed", seed,
                   "--store", &output.to_string_lossy(),
                   "-o", &tmp.path().join("traj.tsv").to_string_lossy()])
            .status().expect("spawn");
    }

    // `camdl list` should find both
    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["list", "--store", &output.to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success(), "list should succeed");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("seed_42"), "list should include seed_42: {}", stdout);
    assert!(stdout.contains("seed_99"), "list should include seed_99: {}", stdout);
    assert!(stdout.contains("baseline"), "list should show scenario name");
}

/// `--starts-from` (2026-05-25 CLI UX rev 2) and `--init from_mle --mle`
/// (the `[stages]` → `[method]` split) are both gone. Users get an actionable
/// error naming the replacement, and `--starts from_mle=<hash>` resolves a
/// short prefix of a method leaf's `run_id` to that leaf (Hardening #9's
/// resolver behaviour under the new spelling).
#[test]
fn starts_from_resolves_short_hash() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let results = tmp.path().join("results");
    // Fake a method dir with a known-hash run.json. The resolver
    // walks results/fits/**, so we need to place the run.json
    // under that structure.
    let method = results.join("fits").join("demo-abc12345")
        .join("if2-1fb03eee").join("seed_1-06cbd6b3");
    std::fs::create_dir_all(&method).unwrap();
    // gh#147: the resolver matches a `FitMethod` `runid::RunRecord` leaf on its
    // `run_id` hex prefix. Plant a leaf whose `run_id` starts `deadbeef`.
    let target_hash = "deadbeefc0ffee00000000000000000000000000000000000000000000000000";
    let run_json = format!(r#"{{
        "format_version":1,"kind":"fit_method","run_id":"{}","hash_version":1,"calendar":{{"days_per_unit":1.0,"origin":null,"time_unit":"days"}},
        "ir_version":"0.7","engine_version":"0.1.0+test",
        "levels":[
            {{"name":"fit","label":"demo","hash":"abc1234500000000000000000000000000000000000000000000000000000000","schema_version":1}},
            {{"name":"method","label":"if2","hash":"1fb03eee00000000000000000000000000000000000000000000000000000000","schema_version":1}},
            {{"name":"seed","label":"seed_1","hash":"06cbd6b300000000000000000000000000000000000000000000000000000000","schema_version":1}}
        ],
        "status":"completed","artifacts":{{}},
        "inputs":{{"method":"if2","backend":"chain_binomial","seed":1,"n_chains":2}},
        "provenance":{{"created_at":"2026-04-19T12:00:00Z","argv":[]}}
    }}"#, target_hash);
    std::fs::write(method.join("run.json"), run_json).unwrap();

    // Exercise the short-hash leaf resolver through `fit run`: the removed
    // flags must error with the actionable replacement message, and the
    // `--starts from_mle=<hash>` spelling resolves the same planted leaf. Run
    // from `tmp` so the default `./results` resolver finds the leaf planted
    // above.
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
[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 50
iterations = 3
cooling = 0.7
"#, ir.display(), data.display())).unwrap();

    // Removed flags, every one named with its replacement in one message
    // (the ebola agents' habitual `--stage posterior --init from_prior`).
    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .current_dir(tmp.path())
        .args(["fit", "run", &fit_toml.to_string_lossy(),
               "--stage", "posterior", "--init", "from_prior",
               "--starts-from", "deadbeef"])
        .output().expect("spawn");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(),
        "removed flags must fail with the removed-flag error");
    assert!(stderr.contains("these `camdl fit run` flags were removed"),
        "expected the removed-flag error, got: {}", stderr);
    assert!(stderr.contains("--stage posterior:") && stderr.contains("drop the flag"),
        "--stage must be named as gone, got: {}", stderr);
    assert!(stderr.contains("--init from_prior: write `--starts from_prior`"),
        "--init must name `--starts`, got: {}", stderr);
    assert!(stderr.contains("--starts-from deadbeef: write `--starts from_mle=deadbeef`"),
        "--starts-from must name its replacement, got: {}", stderr);

    // gh#889: `camdl profile` answers the same four flags the same way, and
    // refuses before it does any work — the earlier surface accepted them and
    // ran with a start rule the flag was meant to select.
    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .current_dir(tmp.path())
        .args(["profile", &ir.to_string_lossy(),
               "--data", &format!("cases={}", data.display()),
               "--sweep", "beta=lin(0.1,0.5,2)", "--rw-sd", "auto",
               "--particles", "20",
               "--init", "from_params", "--params", "truth.toml"])
        .output().expect("spawn");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(),
        "profile's removed flags must fail, not be ignored: {stderr}");
    assert!(stderr.contains("these `camdl profile` flags were removed"),
        "expected the removed-flag error naming this verb, got: {stderr}");
    assert!(stderr.contains("--init from_params: write `--starts from_params=<source>`"),
        "--init must name `--starts`, got: {stderr}");
    assert!(stderr.contains("--params truth.toml: write `--starts from_params=truth.toml`"),
        "--params must name its replacement, got: {stderr}");
    assert!(!stderr.contains("compiling"),
        "the refusal must land before any work, got: {stderr}");

    // Bad hash with the new spelling: the resolver finds no leaf and no fit.
    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .current_dir(tmp.path())
        .args(["fit", "run", &fit_toml.to_string_lossy(),
               "--starts", "from_mle=zzzznonexistent"])
        .output().expect("spawn");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "bad hash must fail");
    assert!(stderr.contains("starts = from_mle zzzznonexistent")
            && stderr.contains("no fit found for zzzznonexistent"),
        "expected the handle named and 'no fit found' — a bad handle is refused before \
         the model is loaded, got: {}", stderr);

    // Good hash: resolves to the leaf we planted. Resolution happens before
    // the fit does anything expensive, so the success path is that the
    // failure comes AFTER the lookup — the planted leaf holds no
    // `fit_state.toml`, and the error says so about that leaf.
    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .current_dir(tmp.path())
        .args(["fit", "run", &fit_toml.to_string_lossy(),
               "--starts", "from_mle=deadbeef"])
        .output().expect("spawn");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("no fit found for"),
        "short-hash 'deadbeef' should resolve to the planted leaf, got: {}", stderr);
    assert!(stderr.contains("seed_1-06cbd6b3") && stderr.contains("fit_state.toml"),
        "the planted leaf must be the one read next, got: {}", stderr);
}

/// `camdl list --kind fit` should hide sim rows entirely; `--kind sim`
/// should hide fit rows. Covers docs/dev/notes/2026-04-20-unified-output-tree-cleanup.md:L2.
#[test]
fn list_kind_filter_isolates_sections() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");
    let store = tmp.path().join("store");

    // Cache one sim, in the run store.
    Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline", "--seed", "7",
               "--store", &store.to_string_lossy(),
               "-o", &tmp.path().join("t.tsv").to_string_lossy()])
        .status().expect("spawn");
    // Synthesise a CAS fit (single `mle` method leaf + sidecar), in the
    // output root fits are stored under.
    write_cas_fit(&output, "demo", "abc12345", &["if2"], "m");

    let fit_only = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["list", "--kind", "fit", &output.to_string_lossy(),
               "--store", &store.to_string_lossy()])
        .output().expect("spawn");
    let s = String::from_utf8_lossy(&fit_only.stdout);
    assert!(s.contains("demo"), "--kind fit should show fits: {}", s);
    assert!(!s.contains("seed_7"), "--kind fit must hide sims: {}", s);

    let sim_only = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["list", "--kind", "sim", &output.to_string_lossy(),
               "--store", &store.to_string_lossy()])
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
    // gh#147 (M3.2): a CAS fit segment with two method leaves + the fit-level
    // sidecar. `read_fit_segment` derives one fit entry whose `methods_declared`
    // comes from the leaves (`if2`, `pgas`).
    write_cas_fit(&output, "demo", "abc12345", &["if2", "pgas"], "m000");

    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE")
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
    // One row per run, under the problem folder's line.
    assert!(stdout.lines().any(|l| l.starts_with("problem ") && l.contains("demo.camdl")),
        "a problem line heads the folder's runs: {}", stdout);
    for m in ["if2", "pgas"] {
        assert!(stdout.lines().any(|l| l.split_whitespace().any(|w| w == m)),
            "a row for the {m} run: {}", stdout);
    }
    // These synthetic leaves hold no stored summary: the verdict says so,
    // and the listing still succeeds.
    assert!(stdout.contains("unreadable:"), "an unreadable verdict renders: {}", stdout);
}

/// Tamper-with-artifact regression: a corrupted `traj.tsv` whose bytes no
/// longer match the cell's recorded `files` digests must never be served. A
/// re-run finds the cell damaged, moves it to `.quarantine/` and simulates it
/// again, so the corrupt bytes are replaced with the correct trajectory.
#[test]
fn cas_tampered_leaf_is_repaired_on_rerun() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");

    let run_once = || {
        Command::new(&bin).env_remove("CLICOLOR_FORCE")
            .args(["simulate", &golden_sir_basic().to_string_lossy(),
                   "--scenario", "baseline",
                   "--seed", "42",
                   "--store", &output.to_string_lossy(),
                   "-o", &tmp.path().join("traj.tsv").to_string_lossy()])
            .output().expect("spawn")
    };

    let first = run_once();
    assert!(first.status.success());
    let dir = sole_cell(&output).dir;
    let good = std::fs::read(dir.join("traj.tsv")).unwrap();

    // Corrupt the cached trajectory.
    std::fs::write(dir.join("traj.tsv"), b"tampered-and-shorter").unwrap();

    // The re-run repairs it: the leaf's traj.tsv is back to the correct bytes,
    // never left holding the corrupt content.
    let out = run_once();
    assert!(out.status.success(), "re-run should succeed");
    let repaired = sole_cell(&output).dir;
    assert_eq!(repaired, dir, "the repaired cell is at the same folder");
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
    // gh#147 (M3.2): a CAS fit is addressed by its method-leaf `run_id` prefix;
    // there is no legacy fit-wide record. `run_id` starts `deadbeef` so the
    // short prefix matches.
    let run_id = "deadbeefc0ffee00000000000000000000000000000000000000000000000000";
    write_cas_fit_method(&output, run_id, "demo");
    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["show", "deadbee", "--root", &output.to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success(), "show by run_id prefix should resolve: stderr={}",
        String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("fit_method"), "should render a fit-method: {stdout}");
    assert!(stdout.contains("if2"), "method label missing: {stdout}");
}


#[test]
fn cat_emits_cached_trajectory() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");

    Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline",
               "--seed", "42",
               "--store", &output.to_string_lossy(),
               "-o", &tmp.path().join("traj.tsv").to_string_lossy()])
        .status().expect("spawn");

    // Find the stored cell, derive the run_id short prefix.
    let cell = sole_cell(&output);
    let dir = cell.dir.clone();
    let short = &cell.run_id[..8];

    // `camdl cat <short>` uniquely resolves and emits the TSV
    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["cat", short, "--store", &output.to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success(), "cat short-hash should resolve uniquely");
    let stdout = String::from_utf8_lossy(&out.stdout);
    // The stored cell's trajectory carries no `# camdl <version>` header and
    // no coordinate columns; it starts with the `t` column header.
    assert!(stdout.starts_with("t\t"), "cat should emit the trajectory header row: {:?}",
        &stdout[..stdout.len().min(40)]);
    assert!(stdout.contains("\tS\t") || stdout.contains("\tI\t"),
        "cat should include compartment columns");

    // Cached trajectory bytes should match the stdout of `cat`
    let cached = std::fs::read(dir.join("traj.tsv")).unwrap();
    assert_eq!(out.stdout, cached, "cat output must match cached bytes byte-for-byte");
}

/// A `--sweep` point is recoverable from the store: each cell's identity
/// records the swept value, its params folder is labelled by it
/// (`beta_0.2`), `camdl list` shows it, and no grid-level `manifest.json` is
/// written.
#[test]
fn sweep_point_is_recorded_in_run_json_and_listed() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");

    // The golden sir_basic IR directly (no .camdl → camdlc step), the
    // non-swept params pinned by a params file. No `--scenario`: sir_basic's
    // `baseline` PRESET sets beta, so a beta sweep under it is refused
    // (`scenario_coordinate_collision_gh572.rs`); the implicit arm leaves beta
    // to the sweep.
    let params_path = tmp.path().join("params.toml");
    std::fs::write(&params_path, "gamma = 0.1\nN0 = 1000\nI0 = 10\n").unwrap();

    let st = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--params", &params_path.to_string_lossy(),
               "--sweep", "beta=0.2,0.3,0.4",
               "--seed", "1",
               "--store", &output.to_string_lossy()])
        .status().expect("spawn");
    assert!(st.success(), "simulate --sweep should succeed");

    // One cell per sweep point × scenario × seed = 3.
    let cells = live_cells(&output);
    assert_eq!(cells.len(), 3, "expected 3 cells for a 3-point sweep");

    // The sweep point is a resolved parameter value (not the scenario delta,
    // not the model hash); the params folder's label encodes it.
    let mut beta_values: Vec<f64> = Vec::new();
    for cell in &cells {
        let beta = cell.params()["beta"].as_f64().expect("beta in the identity's params");
        let label = cell.params_dir_name(); // e.g. "beta_0.2"
        let from_label: f64 = label.strip_prefix("beta_")
            .unwrap_or_else(|| panic!("params label should be beta_<v>, got {label}"))
            .parse().expect("beta value parses");
        assert_eq!(from_label, beta, "the params folder is labelled by the swept value");
        beta_values.push(beta);
    }
    beta_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!((beta_values[0] - 0.2).abs() < 1e-9);
    assert!((beta_values[1] - 0.3).abs() < 1e-9);
    assert!((beta_values[2] - 0.4).abs() < 1e-9);

    assert!(sim_store_cells::walk_files(&output).iter()
            .all(|p| p.file_name().is_none_or(|n| n != "manifest.json")),
        "a sweep must not write a manifest.json");

    // `camdl list --kind sim` shows each cell's beta value.
    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["list", "--store", &output.to_string_lossy(), "--kind", "sim"])
        .output().expect("spawn");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("beta_0.2"), "list should show beta_0.2: {}", stdout);
    assert!(stdout.contains("beta_0.3"), "list should show beta_0.3: {}", stdout);
    assert!(stdout.contains("beta_0.4"), "list should show beta_0.4: {}", stdout);
}

/// `--dry-run` on a sweep prints the resolved grid on stderr, exits 0, and
/// touches nothing under the store.
#[test]
fn sweep_dry_run_prints_the_grid_and_writes_nothing() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");
    let params_path = tmp.path().join("params.toml");
    std::fs::write(&params_path, "gamma = 0.1\nN0 = 1000\nI0 = 10\n").unwrap();

    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--params", &params_path.to_string_lossy(),
               "--sweep", "beta=0.2,0.3,0.4",
               "--seed", "1",
               "--store", &output.to_string_lossy(),
               "--dry-run"])
        .output().expect("spawn");
    assert!(out.status.success(), "dry-run should exit 0");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("camdl simulate (dry run)"),
        "stderr should mark the dry run: {}", stderr);
    assert!(stderr.contains("sweep: beta = [0.2, 0.3, 0.4]"),
        "stderr should include the sweep grid: {}", stderr);
    assert!(stderr.contains("(dry run — no simulation, no files written.)"),
        "stderr should confirm no simulation ran: {}", stderr);

    // Must not have written any run files.
    assert!(sim_store_cells::walk_files(&output).iter()
            .all(|p| p.file_name().is_none_or(|n| n != "run.json")),
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

    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE")
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

    Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline",
               "--seed", "42",
               "--store", &output.to_string_lossy(),
               "-o", &tmp.path().join("traj.tsv").to_string_lossy()])
        .status().expect("spawn");

    let cell = sole_cell(&output);
    let short = &cell.run_id[..8];

    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["show", short, "--store", &output.to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success(), "show should succeed");
    let stdout = String::from_utf8_lossy(&out.stdout);
    // Check that show emits the key fields
    assert!(stdout.contains("baseline"), "should show scenario");
    assert!(stdout.contains("42"), "should show seed");
    assert!(stdout.contains("chain_binomial"), "should show backend (config level)");
}

/// Test (4a from review): `camdl show <fit-method-hash>` resolves and
/// renders the FitMethod payload. Pre-show-coverage-collapse, this
/// returned "unrecognised kind".
/// A content-addressed fit-method leaf (`runid::RunRecord`, M3.2) at
/// `fits/{fit}/{method}/{seed}/run.json`, addressable by `run_id` prefix.
fn write_cas_fit_method(output: &Path, run_id: &str, fit_label: &str) -> PathBuf {
    let leaf = output.join("fits")
        .join(format!("{fit_label}-5ca1ab1e"))
        .join("if2-1fb03eee")
        .join("seed_42-06cbd6b3");
    std::fs::create_dir_all(&leaf).unwrap();
    let rec = format!(r#"{{
        "format_version": 1,
        "kind": "fit_method",
        "run_id": "{run_id}",
        "hash_version": 1,"calendar":{{"days_per_unit":1.0,"origin":null,"time_unit":"days"}},
        "ir_version": "0.7",
        "engine_version": "0.1.0+test",
        "levels": [
            {{"name":"fit","label":"{fit_label}","hash":"5ca1ab1e00000000000000000000000000000000000000000000000000000000","schema_version":1}},
            {{"name":"method","label":"if2","hash":"1fb03eee00000000000000000000000000000000000000000000000000000000","schema_version":1}},
            {{"name":"seed","label":"seed_42","hash":"06cbd6b300000000000000000000000000000000000000000000000000000000","schema_version":1}}
        ],
        "status": "completed",
        "artifacts": {{}},
        "inputs": {{"method":"if2","backend":"chain_binomial","seed":42,"n_chains":4,"best_loglik":-123.45,"best_chain":1}},
        "provenance": {{"created_at":"2026-04-30T12:00:00Z","argv":["camdl","fit","run"]}}
    }}"#);
    std::fs::write(leaf.join("run.json"), rec).unwrap();
    leaf
}

/// A store written before the run.json kind `fit_stage` became `fit_method`
/// is not read, and the readers say so. Before, `list` printed "(no cached
/// runs)" and `show` "could not read run.json", so a user holding an old store
/// could not tell it from an empty one.
#[test]
fn a_leaf_under_the_old_fit_kind_is_named_by_list_and_show() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");
    let leaf = write_cas_fit_method(&output, &"ab".repeat(32), "old");
    let rj = leaf.join("run.json");
    let text = std::fs::read_to_string(&rj).unwrap();
    std::fs::write(&rj, text.replace(r#""kind": "fit_method""#, r#""kind": "fit_stage""#)).unwrap();

    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE").args(["list", &output.to_string_lossy()]).output().expect("spawn");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("fit_stage") && stderr.contains("fit_method")
            && stderr.contains("run the fit again"),
        "list must name the renamed kind and the remedy: stderr={stderr}");
    assert!(!String::from_utf8_lossy(&out.stdout).contains("old-5ca1ab1e"),
        "the old leaf is not listed as a run");

    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE").args(["show", &leaf.to_string_lossy()]).output().expect("spawn");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "show must refuse the old record");
    assert!(stderr.contains("fit_stage") && stderr.contains("fit_method"),
        "show must name the renamed kind: stderr={stderr}");
}

/// Write a content-addressed fit segment `fits/<label>-<fit_h8>/` with one
/// `FitMethod` leaf per method (`<method>-<h8>/seed_1-<h8>/run.json`) plus the
/// fit-level sidecar (`fit.meta.json` — the label + model-identity home). This is
/// the shape `read_fit_segment` derives a single fit-level entry from, so
/// `list` / `fit table` see one fit with `methods_declared` taken from the
/// leaves. Returns the segment dir.
fn write_cas_fit(output: &Path, label: &str, fit_h8: &str, methods: &[&str], model_identity: &str) -> PathBuf {
    let seg = output.join("fits").join(format!("{label}-{fit_h8}"));
    std::fs::create_dir_all(&seg).unwrap();
    let fit_hash = format!("{fit_h8}{}", "0".repeat(64 - fit_h8.len()));
    for (i, method) in methods.iter().enumerate() {
        let nn = i + 1;
        let leaf = seg
            .join(format!("{method}-1fb03eee"))
            .join("seed_1-06cbd6b3");
        std::fs::create_dir_all(&leaf).unwrap();
        // Any distinct 64-hex run_id per leaf; `read_fit_segment` reads the
        // `fit`-level hash (shared) for the fit-level entry's `run.hash`.
        let run_id = format!("{:0<64}", format!("abc{nn}"));
        let rec = format!(r#"{{
            "format_version": 1,
            "kind": "fit_method",
            "run_id": "{run_id}",
            "hash_version": 1,"calendar":{{"days_per_unit":1.0,"origin":null,"time_unit":"days"}},
            "ir_version": "0.7",
            "engine_version": "0.1.0+test",
            "levels": [
                {{"name":"fit","label":"{label}","hash":"{fit_hash}","schema_version":1}},
                {{"name":"method","label":"{method}","hash":"1fb03eee00000000000000000000000000000000000000000000000000000000","schema_version":1}},
                {{"name":"seed","label":"seed_1","hash":"06cbd6b300000000000000000000000000000000000000000000000000000000","schema_version":1}}
            ],
            "status": "completed",
            "artifacts": {{}},
            "inputs": {{"method":"{method}","backend":"chain_binomial","seed":1,"n_chains":2}},
            "provenance": {{"created_at":"2026-04-19T12:00:00Z","argv":["camdl","fit","run"]}}
        }}"#);
        std::fs::write(leaf.join("run.json"), rec).unwrap();
    }
    std::fs::write(
        seg.join("fit.meta.json"),
        format!(r#"{{"model_identity":"{model_identity}","source_model":"demo.camdl","schema_version":3,"calendar":{{"days_per_unit":1.0,"origin":null,"time_unit":"days"}},"source_config":"demo.toml"}}"#),
    )
    .unwrap();
    seg
}

/// Write a fit segment holding two leaves of the SAME algorithm that differ
/// only in their `method`-level hash — the shape a store takes when one fit is
/// run twice at different hyperparameters (two PGAS leaves, 12 sweeps vs 16).
/// Returns the segment dir and the two method hashes, in on-disk order.
fn write_two_leaves_of_one_method(output: &Path, method: &str) -> (PathBuf, [String; 2]) {
    let seg = output.join("fits").join("twin-abc12345");
    std::fs::create_dir_all(&seg).unwrap();
    let fit_hash = format!("abc12345{}", "0".repeat(56));
    let hashes = [
        format!("34f379b8{}", "0".repeat(56)),
        format!("d5c282ce{}", "0".repeat(56)),
    ];
    for (i, method_hash) in hashes.iter().enumerate() {
        let leaf = seg
            .join(format!("{method}-{}", &method_hash[..8]))
            .join("seed_1-06cbd6b3");
        std::fs::create_dir_all(&leaf).unwrap();
        let run_id = format!("{:0<64}", format!("beef{}", i + 1));
        let rec = format!(r#"{{
            "format_version": 1,
            "kind": "fit_method",
            "run_id": "{run_id}",
            "hash_version": 1,"calendar":{{"days_per_unit":1.0,"origin":null,"time_unit":"days"}},
            "ir_version": "0.7",
            "engine_version": "0.1.0+test",
            "levels": [
                {{"name":"fit","label":"twin","hash":"{fit_hash}","schema_version":1}},
                {{"name":"method","label":"{method}","hash":"{method_hash}","schema_version":1}},
                {{"name":"seed","label":"seed_1","hash":"06cbd6b300000000000000000000000000000000000000000000000000000000","schema_version":1}}
            ],
            "status": "completed",
            "artifacts": {{}},
            "inputs": {{"method":"{method}","backend":"chain_binomial","seed":1,"n_chains":2}},
            "provenance": {{"created_at":"2026-04-19T12:00:00Z","argv":["camdl","fit","run"]}}
        }}"#);
        std::fs::write(leaf.join("run.json"), rec).unwrap();
    }
    std::fs::write(
        seg.join("fit.meta.json"),
        r#"{"model_identity":"m000","source_model":"demo.camdl","schema_version":3,"calendar":{"days_per_unit":1.0,"origin":null,"time_unit":"days"},"source_config":"demo.toml"}"#,
    )
    .unwrap();
    (seg, hashes)
}

/// gh#895 + gh#901: `camdl list --kind fit --format json` speaks *method*, and
/// its per-leaf entries are distinguishable from one another.
///
/// Two things were wrong with the row. The keys were `methods`,
/// `stages_declared` and per-entry `method` — a vocabulary the store levels, the
/// fit config and the CLI had all left. And two leaves of one algorithm in one
/// segment were indistinguishable: each carries its own `method`-level hash and
/// its own `run_id`, and the row carried neither, so a reader could not say
/// which entry was which leaf or address either one.
#[test]
fn fit_list_json_names_methods_and_can_tell_two_leaves_apart() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");
    let (_seg, hashes) = write_two_leaves_of_one_method(&output, "pgas");

    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["list", "--kind", "fit", "--format", "json", &output.to_string_lossy()])
        .output()
        .expect("spawn");
    assert!(out.status.success(), "list failed: {}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let rows: Vec<serde_json::Value> = text.lines()
        .filter(|l| l.contains("\"kind\":\"fit\""))
        .map(|l| serde_json::from_str(l).expect("each row is JSON"))
        .collect();

    // v3: one row per run, each tagged, each naming its own run and path.
    assert_eq!(rows.len(), 2, "both leaves are listed, one row each: {text}");
    for row in &rows {
        assert_eq!(row["schema"]["name"], "fit_list_row", "{row}");
        assert_eq!(row["schema"]["version"], 3, "{row}");
        assert_eq!(row["method"], "pgas", "{row}");
        assert_eq!(row["problem"]["config"], "demo.toml", "{row}");
        assert_eq!(row["problem"]["model"], "demo.camdl", "{row}");
        assert!(row["problem"]["fit_id"].as_str().unwrap().starts_with("abc12345"), "{row}");
        assert_eq!(row["seed"], 1, "{row}");
        assert_eq!(row["status"], "completed", "{row}");
        assert_eq!(row["verdict"]["kind"], "unreadable", "{row}");
        assert!(row.get("methods").is_none(), "no v2 `methods` array: {row}");
        let p = row["path"].as_str().expect("each row carries a path");
        assert!(Path::new(p).join("run.json").is_file(), "the path names the leaf: {p}");
    }
    let ids: std::collections::BTreeSet<&str> = rows.iter()
        .map(|r| r["run_id"].as_str().expect("run_id is a string"))
        .collect();
    assert_eq!(ids.len(), 2, "two distinct run_ids: {text}");
    let paths: Vec<&str> = rows.iter().map(|r| r["path"].as_str().unwrap()).collect();
    for h in &hashes {
        assert!(paths.iter().any(|p| p.contains(&format!("pgas-{}", &h[..8]))),
            "a row for the leaf of method hash {h}: {paths:?}");
    }
}

/// gh#147 (M3.2): `camdl show <fit-method path | run_id prefix>` renders the
/// CAS fit-method `RunRecord` — factored levels + the recorded method metadata.
#[test]
fn show_renders_fit_method_metadata() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");
    let run_id = "5ca1ab1e00000000000000000000000000000000000000000000000000000000";
    let leaf = write_cas_fit_method(&output, run_id, "parent");

    // Resolve by full path.
    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["show", &leaf.to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success(), "show fit-method failed: stderr={}",
        String::from_utf8_lossy(&out.stderr));
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("fit_method"), "kind label missing: {}", s);
    assert!(s.contains("if2"),       "method name missing: {}", s);
    assert!(s.contains("if2"),       "method missing: {}", s);
    assert!(s.contains("-123.45"),   "best_loglik missing: {}", s);

    // Resolve by run_id prefix.
    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["show", "5ca1ab1e", "--root", &output.to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success(), "show by run_id prefix failed: stderr={}",
        String::from_utf8_lossy(&out.stderr));
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("if2"));
}


/// A cell stored by a later invocation is found by `show` by its run-id
/// prefix, as the first one is.
#[test]
fn out_of_band_leaf_is_found_via_walk_fallback() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");

    // Store one sim, then a SECOND (seed 2) in a later invocation.
    for seed in ["1", "2"] {
        Command::new(&bin).env_remove("CLICOLOR_FORCE")
            .args(["simulate", &golden_sir_basic().to_string_lossy(),
                   "--scenario", "baseline", "--seed", seed,
                   "--store", &output.to_string_lossy(),
                   "-o", &tmp.path().join("t.tsv").to_string_lossy()])
            .status().expect("spawn");
    }

    // The later cell's run_id.
    let later = live_cells(&output).into_iter().find(|c| c.seed() == 2).expect("seed 2 cell");
    let later_short = &later.run_id[..8];

    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["show", later_short, "--store", &output.to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success(),
        "a later cell must resolve: stderr={}",
        String::from_utf8_lossy(&out.stderr));
    assert!(!String::from_utf8_lossy(&out.stderr).contains("no run matches"),
        "must not report 'no match' for a later cell");
}

/// A cell whose folder was removed must NOT resolve to the dead path; `show`
/// reports a clean "no match" instead of pointing at the missing cell.
#[test]
fn removed_leaf_does_not_resolve_to_dead_path() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("output");

    Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline", "--seed", "1",
               "--store", &output.to_string_lossy(),
               "-o", &tmp.path().join("t.tsv").to_string_lossy()])
        .status().expect("spawn");

    let cell = sole_cell(&output);
    let short = cell.run_id[..8].to_string();
    // It resolves while present.
    assert!(Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["show", &short, "--store", &output.to_string_lossy()])
        .status().expect("spawn").success(), "resolves while present");

    // Remove the cell out of band; the store's index still names it.
    std::fs::remove_dir_all(&cell.dir).unwrap();

    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["show", &short, "--store", &output.to_string_lossy()])
        .output().expect("spawn");
    assert!(!out.status.success(),
        "a removed cell must not resolve (no dead path)");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("no run matches"),
        "expected clean 'no run matches', got: {}", stderr);
}

// ─── Combined table (multi-cell) tests ──────────────────────────────────────

/// Round-trip: a 3-replicate run under `--label` stores 3 cells; `cat @LABEL`
/// emits the combined wide-format TSV (with a `replicate` column, 3
/// replicates), byte-identical to the `-o` export.
#[test]
fn ensemble_round_trip_deps_cat_and_list() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("out");
    let mirror = tmp.path().join("combined.tsv");

    let st = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline", "--seed", "42", "--replicates", "3",
               "--store", &out.to_string_lossy(), "--label", "three",
               "-o", &mirror.to_string_lossy()])
        .status().expect("spawn");
    assert!(st.success(), "3-replicate simulate should succeed");

    assert_eq!(live_cells(&out).len(), 3, "expected 3 cells");

    // `cat @three` emits the combined TSV byte-identical to the `-o` export,
    // and that TSV has a `replicate` column with 3 replicates.
    let cat = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["cat", "@three", "--store", &out.to_string_lossy()])
        .output().expect("spawn");
    assert!(cat.status.success(), "cat @three failed: {}",
        String::from_utf8_lossy(&cat.stderr));
    let mirror_bytes = std::fs::read(&mirror).unwrap();
    assert!(!mirror_bytes.is_empty(), "the -o export must be non-empty");
    assert_eq!(cat.stdout, mirror_bytes,
        "cat @LABEL must equal the -o combined TSV byte-for-byte");

    // The combined TSV header carries a `replicate` column, and the data has
    // exactly 3 distinct replicate values, counted from 0 (one index base for
    // every user-facing index, proposal 2026-10-24-batch-to-simulate §3).
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
    assert_eq!(reps, ["0", "1", "2"].into_iter().collect(),
        "combined TSV must contain replicates 0,1,2, got {reps:?}");
}

/// Item C: `simulate` (single AND multi-cell) with NO `-o` writes NOTHING to
/// stdout — the stored cells are the system of record. The `-o` export still
/// works, and the cells exist either way.
#[test]
fn simulate_writes_nothing_to_stdout_without_output_flag() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();

    // Single-run, no -o: empty stdout, one cell.
    let out1 = tmp.path().join("out1");
    let single = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline", "--seed", "42",
               "--store", &out1.to_string_lossy()])
        .output().expect("spawn");
    assert!(single.status.success(), "single-run simulate (no -o) should succeed: {}",
        String::from_utf8_lossy(&single.stderr));
    assert!(single.stdout.is_empty(),
        "single-run simulate with no -o must write NOTHING to stdout, got {} bytes",
        single.stdout.len());
    assert_eq!(live_cells(&out1).len(), 1, "the cell must still be written");

    // Multi-cell, no -o: empty stdout, but the cells exist.
    let out3 = tmp.path().join("out3");
    let multi = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline", "--seed", "42", "--replicates", "3",
               "--store", &out3.to_string_lossy()])
        .output().expect("spawn");
    assert!(multi.status.success(), "multi-cell simulate (no -o) should succeed: {}",
        String::from_utf8_lossy(&multi.stderr));
    assert!(multi.stdout.is_empty(),
        "multi-cell simulate with no -o must write NOTHING to stdout, got {} bytes",
        multi.stdout.len());
    assert_eq!(live_cells(&out3).len(), 3, "the 3 cells must still be written");

    // With -o, the export is written and matches `cat @LABEL`.
    let out3b = tmp.path().join("out3b");
    let mirror = tmp.path().join("m.tsv");
    let mirrored = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline", "--seed", "42", "--replicates", "3",
               "--store", &out3b.to_string_lossy(), "--label", "m",
               "-o", &mirror.to_string_lossy()])
        .output().expect("spawn");
    assert!(mirrored.status.success(), "multi-cell simulate (-o) should succeed");
    assert!(mirrored.stdout.is_empty(),
        "even WITH -o, simulate writes the trajectory to the file, not stdout");
    let mirror_bytes = std::fs::read(&mirror).unwrap();
    let cat = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["cat", "@m", "--store", &out3b.to_string_lossy()])
        .output().expect("spawn");
    assert!(cat.status.success(), "cat @m: {}", String::from_utf8_lossy(&cat.stderr));
    assert_eq!(mirror_bytes, cat.stdout,
        "the -o export must be byte-identical to the table `cat @LABEL` renders");
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

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
    let compiled = Command::new(bin).env_remove("CLICOLOR_FORCE")
        .args(["dev", "compile", model.to_str().unwrap(), "-o", ir.to_str().unwrap()])
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().expect("spawn compile");
    if !compiled.status.success() {
        eprintln!("skipping: camdl compile failed (camdlc unavailable): {}",
            String::from_utf8_lossy(&compiled.stderr));
        return None;
    }
    Some(ir)
}

/// `simulate --event-log` records the event log as a by-product of the cell,
/// `<cell>/event_log/event_log-<hash>/event_log.tsv`. The recorder is passive
/// (Tier 2a), so the cell's run_id, `traj.tsv` bytes and own files are those of
/// a plain `simulate` at the same seed. `--event-log PATH` also writes PATH as
/// a byte-identical export (symmetric with `-o` for the trajectory).
#[test]
fn event_log_lands_in_the_sim_leafs_child() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let Some(ir) = compile_lineage_model(&bin, tmp.path()) else { return; };
    let ir_s = ir.to_string_lossy().into_owned();
    let common = ["--backend", "gillespie", "--seed", "7",
                  "--param", "beta=0.6", "--param", "gamma=0.2", "--param", "N0=500"];

    // (1) plain simulate → cell with traj.tsv only.
    let out_plain = tmp.path().join("plain");
    let st = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["simulate", &ir_s]).args(common)
        .args(["--store", &out_plain.to_string_lossy(),
               "-o", &tmp.path().join("traj_plain.tsv").to_string_lossy()])
        .status().expect("spawn");
    assert!(st.success(), "plain simulate should succeed");
    let plain = sole_cell(&out_plain);
    let plain_traj = std::fs::read(plain.dir.join("traj.tsv")).unwrap();
    let plain_files: Vec<String> =
        plain.run_json()["files"].as_object().unwrap().keys().cloned().collect();
    assert!(!plain.dir.join("event_log").exists(),
        "plain simulate must NOT write an event log");

    // (2) simulate --event-log PATH → the same cell files, run_id and traj
    //     bytes, plus the by-product; PATH is a byte-identical export.
    let out_el = tmp.path().join("el");
    let mirror = tmp.path().join("mirror.tsv");
    let st = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["simulate", &ir_s]).args(common)
        .args(["--store", &out_el.to_string_lossy(),
               "--event-log", &mirror.to_string_lossy()])
        .status().expect("spawn");
    assert!(st.success(), "simulate --event-log should succeed");
    let el = sole_cell(&out_el);

    assert_eq!(el.run_id, plain.run_id,
        "--event-log must not change the sim run_id (passive recorder, Tier 2a)");
    assert_eq!(std::fs::read(el.dir.join("traj.tsv")).unwrap(), plain_traj,
        "trajectory must be byte-identical with/without --event-log");
    let el_files: Vec<String> = el.run_json()["files"].as_object().unwrap().keys().cloned().collect();
    assert_eq!(el_files, plain_files, "--event-log leaves the cell's own files as a plain run's");
    let logs = el.by_products("event_log.tsv");
    assert_eq!(logs.len(), 1, "one event log by-product: {logs:?}");
    assert!(logs[0].starts_with(el.dir.join("event_log")),
        "the event log is under the cell's event_log/: {}", logs[0].display());
    assert_eq!(std::fs::read(&mirror).unwrap(), std::fs::read(&logs[0]).unwrap(),
        "--event-log PATH export must be byte-identical to the stored event_log.tsv");
}

/// A `--label` passed to `simulate` names the invocation EVEN WHEN its cell is
/// already stored: `cat @LABEL` renders the run's table. A label-less rerun
/// does not take the label away.
#[test]
fn simulate_label_applies_on_cached_leaf() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("store");
    let gold = golden_sir_basic();
    let run = |label: Option<&str>| {
        let mut args = vec!["simulate".to_string(), gold.to_string_lossy().into_owned(),
            "--scenario".into(), "baseline".into(), "--seed".into(), "123".into(),
            "--store".into(), out.to_string_lossy().into_owned()];
        if let Some(l) = label { args.push("--label".into()); args.push(l.into()); }
        assert!(Command::new(&bin).env_remove("CLICOLOR_FORCE").args(&args).status().expect("spawn").success());
    };
    let cat_label = || Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["cat", "@added later", "--store", &out.to_string_lossy()])
        .output().expect("spawn");

    // 1. fresh run, no label → the label names nothing.
    run(None);
    let cell = sole_cell(&out);
    assert!(!cat_label().status.success(), "no --label → `@added later` names nothing");

    // 2. re-run SAME params WITH --label: the stored cell is reused and the
    //    label names the run.
    run(Some("added later"));
    assert_eq!(sole_cell(&out).dir, cell.dir, "the labelled rerun reuses the stored cell");
    let cat = cat_label();
    assert!(cat.status.success(),
        "simulate --label over a stored cell must apply the label: {}",
        String::from_utf8_lossy(&cat.stderr));
    let table = cat.stdout;

    // 3. label-less rerun must PRESERVE the existing label (absence ≠ clear).
    run(None);
    let cat = cat_label();
    assert!(cat.status.success(), "a label-less rerun must not wipe an existing label");
    assert_eq!(cat.stdout, table, "the label still renders the same table");
}

/// `camdl list --store DIR` lists each cell of a multi-replicate `simulate`
/// on its own row, in the default view and under `--kind sim`.
#[test]
fn list_collapses_ensemble_members_and_accepts_root_flag() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let store = tmp.path().join("store");
    let store_s = store.to_string_lossy().into_owned();
    assert!(Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline", "--seed", "5", "--replicates", "3",
               "--store", &store_s])
        .status().expect("spawn").success());

    // JSONL: one record per line on stdout. Parse each line and inspect the
    // TOP-LEVEL `kind`.
    let count_kind = |args: &[&str], kind: &str| -> usize {
        let out = Command::new(&bin).env_remove("CLICOLOR_FORCE").args(args).output().expect("spawn");
        String::from_utf8_lossy(&out.stdout).lines()
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .filter(|v| v["kind"] == kind)
            .count()
    };

    assert_eq!(count_kind(&["list", "--store", &store_s, "--format", "json"], "sim"), 3,
        "the default list must show the 3 replicate cells");
    assert_eq!(count_kind(&["list", "--store", &store_s, "--kind", "sim", "--format", "json"], "sim"), 3,
        "`list --kind sim` must surface all 3 replicate cells");
}

/// The store banner printed after a `simulate` names the stored cell's folder
/// under the `--store` root (a copy-paste-ready path), plus the
/// `camdl cat <run_id>` that reads the run back.
#[test]
fn stored_banner_includes_output_dir_prefix_and_cat_hint() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let store = tmp.path().join("store");
    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline", "--seed", "7",
               "--store", &store.to_string_lossy()])
        .output().expect("spawn");
    let stderr = String::from_utf8_lossy(&out.stderr);
    let cell = sole_cell(&store);
    let banner = format!("stored {}", cell.dir.display());
    assert!(stderr.contains(&banner),
        "the store banner must name the cell folder '{banner}':\n{stderr}");
    assert!(stderr.contains(&format!("camdl cat {}", cell.run_id)),
        "the banner must tell the user how to read the run back:\n{stderr}");
}

/// `--stdout` stores the cell and streams the combined table to stdout — the
/// escape hatch for piping. It prints the table `-o` would write.
#[test]
fn stdout_streams_trajectory_and_skips_the_store() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let store = tmp.path().join("store");
    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline", "--seed", "7",
               "--store", &store.to_string_lossy(), "--stdout"])
        .output().expect("spawn");
    assert!(out.status.success(), "simulate --stdout should succeed");
    let stdout = String::from_utf8_lossy(&out.stdout);
    // A wide-format table on stdout: a `# <version>` comment, then the header
    // (`replicate`, `scenario`, `seed`, then `t` …), then rows.
    assert!(stdout.lines().any(|l| l.starts_with("replicate\tscenario\tseed\tt\t")),
        "stdout should carry the combined table header, got:\n{stdout}");
    assert!(stdout.lines().count() > 3, "stdout should carry trajectory rows");
    // The cell was stored, and stdout is the table `-o` writes from it.
    assert_eq!(live_cells(&store).len(), 1, "--stdout stores the cell");
    let export = tmp.path().join("t.tsv");
    let o = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["simulate", &golden_sir_basic().to_string_lossy(),
               "--scenario", "baseline", "--seed", "7",
               "--store", &store.to_string_lossy(), "-o", &export.to_string_lossy()])
        .output().expect("spawn");
    assert!(o.status.success());
    assert_eq!(out.stdout, std::fs::read(&export).unwrap(), "--stdout prints the table -o writes");
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
        let st = Command::new(&bin).env_remove("CLICOLOR_FORCE")
            .args(["simulate", &golden_sir_basic().to_string_lossy(),
                   "--scenario", "baseline", "--seed", "7", "--replicates", "3",
                   "--store", &tmp.path().join(tag).to_string_lossy(),
                   "-o", &o.to_string_lossy(), "--progress", mode])
            .status().expect("spawn");
        assert!(st.success(), "simulate --progress {mode} should succeed");
        std::fs::read(&o).unwrap()
    };
    assert_eq!(run("none", "n"), run("pretty", "p"),
        "progress mode must not change the trajectory (bars must be RNG-passive)");
}

