//! The compiled-IR cache: a `.camdl` compiled once is reused across separate
//! `camdl` invocations (keyed on the model content + the camdlc/IR version),
//! so camdlc is skipped on a cache hit. Verified with a counting camdlc shim
//! on PATH and a per-test `CAMDL_IR_CACHE_DIR` (so the real user cache is
//! never touched). Editing the model, or `--no-ir-cache`, must recompile.

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn real_camdlc() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe")
}

fn skip_if_unbuilt() -> Option<(PathBuf, PathBuf)> {
    let bin = camdl_bin();
    let cc = real_camdlc();
    if !bin.exists() || !cc.exists() {
        eprintln!("skipping: camdl/camdlc not built");
        return None;
    }
    Some((bin, cc))
}

const SIR: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.001, 5.0]
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 10000]
}
let N = S + I + R
transitions {
  infection : S --> I @ beta * S * I / N
  recovery  : I --> R @ gamma * I
}
init { S = 499  I = 1 }
simulate { from = 0 'days  to = 20 'days }
"#;

/// A 2-patch SIR whose population table is loaded via `read("pop.tsv")` at
/// compile time and baked into the IR. Editing `pop.tsv` (without touching the
/// `.camdl`) must invalidate the cache (gh#260).
const SIR_PATCHES: &str = r#"
time_unit = 'days
dimensions { patch = [north, south] }
compartments { S, I, R }
stratify(by = patch)
tables {
  N0 : patch = read("pop.tsv")
}
parameters {
  beta  : rate        in [0.001, 1.0]
  gamma : rate        in [0.01,  1.0]
  I0    : count       in [1, 100]
}
let N[p in patch] = S[p] + I[p] + R[p]
transitions {
  infection[p in patch] : S[p] --> I[p]  @ beta * S[p] * I[p] / N[p]
  recovery[p in patch]  : I[p] --> R[p]  @ gamma * I[p]
}
init {
  S[p in patch] = N0[p] - I0
  I[p in patch] = I0
}
simulate { from = 0 'days  to = 28 'days }
"#;

const POP_TSV_A: &str = "patch\tN0\nnorth\t50000\nsouth\t30000\n";
const POP_TSV_B: &str = "patch\tN0\nnorth\t99999\nsouth\t30000\n";

/// A camdlc wrapper that appends a line per *compile* (not the
/// `--camdl-version` probe) before exec'ing the real camdlc.
fn counting_shim(dir: &Path, real: &Path, counter: &Path) -> PathBuf {
    let shim = dir.join("camdlc");
    std::fs::write(&shim, format!(
        "#!/bin/sh\ncase \"$1\" in\n  --camdl-version) ;;\n  *) echo x >> '{}' ;;\nesac\nexec '{}' \"$@\"\n",
        counter.display(), real.display())).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut p = std::fs::metadata(&shim).unwrap().permissions();
        p.set_mode(0o755);
        std::fs::set_permissions(&shim, p).unwrap();
    }
    dir.to_path_buf()
}

fn compiles(counter: &Path) -> usize {
    std::fs::read_to_string(counter).map(|s| s.lines().filter(|l| !l.trim().is_empty()).count()).unwrap_or(0)
}

/// Run `simulate <model>` once with the shim ahead on PATH and the cache dir
/// pointed at `cache_dir`. `no_cache` adds `--no-ir-cache`.
fn run_simulate(bin: &Path, shim_dir: &Path, model: &Path, cache_dir: &Path, out: &Path, no_cache: bool) {
    let old_path = std::env::var("PATH").unwrap_or_default();
    let mut args: Vec<&str> = vec![
        "simulate", model.to_str().unwrap(),
        "--backend", "chain_binomial", "--seed", "1",
        "--param", "beta=0.3", "--param", "gamma=0.1", "--param", "N0=1000",
        "--output-dir", out.to_str().unwrap(), "--progress", "none",
    ];
    if no_cache { args.push("--no-ir-cache"); }
    let st = Command::new(bin)
        .args(&args)
        .env("PATH", format!("{}:{}", shim_dir.display(), old_path))
        .env("CAMDL_IR_CACHE_DIR", cache_dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .status().expect("spawn");
    assert!(st.success(), "simulate should succeed");
}

#[test]
fn ir_compiled_once_then_reused_across_runs() {
    let Some((bin, real)) = skip_if_unbuilt() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("sir.camdl");
    std::fs::write(&model, SIR).unwrap();
    let counter = tmp.path().join("compiles.log");
    let shim = counting_shim(tmp.path(), &real, &counter);
    let cache = tmp.path().join("ircache");

    run_simulate(&bin, &shim, &model, &cache, &tmp.path().join("o1"), false);
    assert_eq!(compiles(&counter), 1, "first run compiles once (cache miss)");

    run_simulate(&bin, &shim, &model, &cache, &tmp.path().join("o2"), false);
    assert_eq!(compiles(&counter), 1,
        "second run of the SAME model reuses the cached IR — camdlc must NOT run again");
}

#[test]
fn editing_the_model_invalidates_the_cache() {
    let Some((bin, real)) = skip_if_unbuilt() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("sir.camdl");
    std::fs::write(&model, SIR).unwrap();
    let counter = tmp.path().join("compiles.log");
    let shim = counting_shim(tmp.path(), &real, &counter);
    let cache = tmp.path().join("ircache");

    run_simulate(&bin, &shim, &model, &cache, &tmp.path().join("o1"), false);
    assert_eq!(compiles(&counter), 1);

    // Change the model content (longer horizon) → different key → recompile.
    std::fs::write(&model, SIR.replace("to = 20 'days", "to = 40 'days")).unwrap();
    run_simulate(&bin, &shim, &model, &cache, &tmp.path().join("o2"), false);
    assert_eq!(compiles(&counter), 2, "an edited model must recompile (content is in the key)");
}

/// Run `simulate` on the 2-patch model (params differ from `SIR`; `N0` is a
/// read() table, not a CLI param).
fn run_patches(bin: &Path, shim_dir: &Path, model: &Path, cache_dir: &Path, out: &Path) {
    let old_path = std::env::var("PATH").unwrap_or_default();
    let st = Command::new(bin)
        .args([
            "simulate", model.to_str().unwrap(),
            "--backend", "chain_binomial", "--seed", "1",
            "--param", "beta=0.3", "--param", "gamma=0.1", "--param", "I0=5",
            "--output-dir", out.to_str().unwrap(), "--progress", "none",
        ])
        .env("PATH", format!("{}:{}", shim_dir.display(), old_path))
        .env("CAMDL_IR_CACHE_DIR", cache_dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .status().expect("spawn");
    assert!(st.success(), "simulate (patches) should succeed");
}

/// gh#260: a file loaded via `read()` is a compile input. Editing `pop.tsv`
/// without touching the `.camdl` must invalidate the cache and recompile —
/// otherwise camdl silently serves IR built from the stale populations.
#[test]
fn editing_a_read_loaded_file_invalidates_the_cache() {
    let Some((bin, real)) = skip_if_unbuilt() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("patches.camdl");
    std::fs::write(&model, SIR_PATCHES).unwrap();
    let pop = tmp.path().join("pop.tsv");
    std::fs::write(&pop, POP_TSV_A).unwrap();
    let counter = tmp.path().join("compiles.log");
    let shim = counting_shim(tmp.path(), &real, &counter);
    let cache = tmp.path().join("ircache");

    run_patches(&bin, &shim, &model, &cache, &tmp.path().join("o1"));
    assert_eq!(compiles(&counter), 1, "first run compiles once (cache miss)");

    run_patches(&bin, &shim, &model, &cache, &tmp.path().join("o2"));
    assert_eq!(compiles(&counter), 1, "unchanged read() file → cache hit, no recompile");

    // Edit the read()-loaded table; the .camdl is byte-identical.
    std::fs::write(&pop, POP_TSV_B).unwrap();

    run_patches(&bin, &shim, &model, &cache, &tmp.path().join("o3"));
    assert_eq!(compiles(&counter), 2,
        "editing the read()-loaded pop.tsv must invalidate the cache and recompile");

    run_patches(&bin, &shim, &model, &cache, &tmp.path().join("o4"));
    assert_eq!(compiles(&counter), 2,
        "re-run after the edit hits the cache (now keyed to the new pop.tsv)");
}

/// gh#260: correctness under the GLOBAL shared cache. Two byte-identical
/// `.camdl` files (→ identical cache key) in different directories with
/// DIFFERENT `pop.tsv` contents must not share an IR entry — the read()-inputs
/// are re-resolved against the *current* model's directory and re-hashed, so B
/// recompiles rather than serving A's IR built from A's populations.
#[test]
fn same_model_different_read_data_do_not_share_a_cache_entry() {
    let Some((bin, real)) = skip_if_unbuilt() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let shared_cache = tmp.path().join("ircache"); // ONE global cache for both
    let counter = tmp.path().join("compiles.log");
    let shim = counting_shim(tmp.path(), &real, &counter);

    // Context A: model + popA.
    let dir_a = tmp.path().join("a");
    std::fs::create_dir_all(&dir_a).unwrap();
    let model_a = dir_a.join("patches.camdl");
    std::fs::write(&model_a, SIR_PATCHES).unwrap();
    std::fs::write(dir_a.join("pop.tsv"), POP_TSV_A).unwrap();

    // Context B: BYTE-IDENTICAL model, a DIFFERENT pop.tsv.
    let dir_b = tmp.path().join("b");
    std::fs::create_dir_all(&dir_b).unwrap();
    let model_b = dir_b.join("patches.camdl");
    std::fs::write(&model_b, SIR_PATCHES).unwrap();
    std::fs::write(dir_b.join("pop.tsv"), POP_TSV_B).unwrap();

    run_patches(&bin, &shim, &model_a, &shared_cache, &tmp.path().join("oa"));
    assert_eq!(compiles(&counter), 1, "context A compiles once (cold cache)");

    run_patches(&bin, &shim, &model_b, &shared_cache, &tmp.path().join("ob"));
    assert_eq!(compiles(&counter), 2,
        "same model bytes but different pop.tsv → must recompile, not reuse A's IR");
}

/// Like `run_simulate` but with `CAMDL_NO_CONSTANT_FOLD` set, so camdlc emits
/// the unfolded IR — a different compile output for the same model.
fn run_simulate_fold_off(bin: &Path, shim_dir: &Path, model: &Path, cache_dir: &Path, out: &Path) {
    let old_path = std::env::var("PATH").unwrap_or_default();
    let st = Command::new(bin)
        .args([
            "simulate", model.to_str().unwrap(),
            "--backend", "chain_binomial", "--seed", "1",
            "--param", "beta=0.3", "--param", "gamma=0.1", "--param", "N0=1000",
            "--output-dir", out.to_str().unwrap(), "--progress", "none",
        ])
        .env("PATH", format!("{}:{}", shim_dir.display(), old_path))
        .env("CAMDL_IR_CACHE_DIR", cache_dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_NO_CONSTANT_FOLD", "1")
        .status().expect("spawn");
    assert!(st.success(), "simulate (fold off) should succeed");
}

#[test]
fn toggling_constant_fold_invalidates_the_cache() {
    let Some((bin, real)) = skip_if_unbuilt() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("sir.camdl");
    std::fs::write(&model, SIR).unwrap();
    let counter = tmp.path().join("compiles.log");
    let shim = counting_shim(tmp.path(), &real, &counter);
    let cache = tmp.path().join("ircache");

    // Fold ON (the default): compile + cache.
    run_simulate(&bin, &shim, &model, &cache, &tmp.path().join("o1"), false);
    assert_eq!(compiles(&counter), 1, "fold-on: first run compiles once");

    // Same model, but CAMDL_NO_CONSTANT_FOLD now set → camdlc emits a DIFFERENT
    // (unfolded) IR. The flag is in the cache key, so this must recompile, not
    // serve the folded variant.
    run_simulate_fold_off(&bin, &shim, &model, &cache, &tmp.path().join("o2"));
    assert_eq!(compiles(&counter), 2, "toggling CAMDL_NO_CONSTANT_FOLD recompiles (flag in key)");

    // The two variants are SEPARATE entries, not an overwrite: fold-off reuses
    // its own entry, and fold-on still hits its original one.
    run_simulate_fold_off(&bin, &shim, &model, &cache, &tmp.path().join("o3"));
    assert_eq!(compiles(&counter), 2, "fold-off reuses its own cache entry");
    run_simulate(&bin, &shim, &model, &cache, &tmp.path().join("o4"), false);
    assert_eq!(compiles(&counter), 2, "fold-on still hits its original entry");
}

#[test]
fn no_ir_cache_flag_bypasses_the_cache() {
    let Some((bin, real)) = skip_if_unbuilt() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("sir.camdl");
    std::fs::write(&model, SIR).unwrap();
    let counter = tmp.path().join("compiles.log");
    let shim = counting_shim(tmp.path(), &real, &counter);
    let cache = tmp.path().join("ircache");

    run_simulate(&bin, &shim, &model, &cache, &tmp.path().join("o1"), true);
    run_simulate(&bin, &shim, &model, &cache, &tmp.path().join("o2"), true);
    assert_eq!(compiles(&counter), 2, "--no-ir-cache must recompile every run");
}

/// Spawn `simulate <model>` WITHOUT waiting — for the concurrency test. The
/// returned `Child` is `wait()`ed by the caller after all peers are launched,
/// so the processes genuinely race on the cold cache.
fn spawn_simulate(bin: &Path, shim_dir: &Path, model: &Path, cache_dir: &Path, out: &Path, seed: &str)
    -> std::process::Child
{
    let old_path = std::env::var("PATH").unwrap_or_default();
    Command::new(bin)
        .args([
            "simulate", model.to_str().unwrap(),
            "--backend", "chain_binomial", "--seed", seed,
            "--param", "beta=0.3", "--param", "gamma=0.1", "--param", "N0=1000",
            "--output-dir", out.to_str().unwrap(), "--progress", "none",
        ])
        .env("PATH", format!("{}:{}", shim_dir.display(), old_path))
        .env("CAMDL_IR_CACHE_DIR", cache_dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        // Discard the TSV so a full pipe buffer can't stall a worker.
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn().expect("spawn")
}

/// gh#214: N concurrent `simulate` of the SAME model on a COLD cache must
/// compile camdlc exactly ONCE (single-flight), not N times. Before the fix
/// every worker missed the cache and spawned its own ~11 GB camdlc → OOM storm;
/// the reproduction counted N invocations. With the single-flight lock one
/// worker compiles and publishes the IR while the rest wait and serve it.
///
/// The assertion is on the camdlc *invocation count* (via the counting shim),
/// which is timing-independent — it does not depend on how the N processes
/// interleave, only that they all contend on one cold key.
#[test]
fn concurrent_simulate_compiles_camdlc_once() {
    let Some((bin, real)) = skip_if_unbuilt() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("sir.camdl");
    std::fs::write(&model, SIR).unwrap();
    let counter = tmp.path().join("compiles.log");
    let shim = counting_shim(tmp.path(), &real, &counter);
    // COLD cache: a dir that does not yet exist — every worker misses.
    let cache = tmp.path().join("ircache");

    const N: usize = 6;
    let children: Vec<_> = (0..N)
        .map(|i| spawn_simulate(
            &bin, &shim, &model, &cache,
            &tmp.path().join(format!("w{i}")), &i.to_string()))
        .collect();

    for (i, mut c) in children.into_iter().enumerate() {
        let st = c.wait().expect("wait");
        assert!(st.success(), "worker {i} should succeed (exit 0)");
    }

    // The single-flight lock dedupes the compile: exactly one camdlc run.
    assert_eq!(
        compiles(&counter), 1,
        "{N} concurrent cold-cache simulates must compile camdlc ONCE (single-flight), \
         not once-per-worker (the gh#214 storm)");

    // The lock file must not linger after the leader finished — it is removed
    // on guard drop so the entry is a clean cache hit thereafter.
    let leftover_locks: Vec<_> = std::fs::read_dir(&cache).unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "lock"))
        .collect();
    assert!(leftover_locks.is_empty(),
        "single-flight .lock must be removed after compile, found: {leftover_locks:?}");

    // A subsequent run is a pure cache hit: no new compile.
    run_simulate(&bin, &shim, &model, &cache, &tmp.path().join("after"), false);
    assert_eq!(compiles(&counter), 1,
        "a warm run after the concurrent storm must hit the cache (0 new compiles)");
}

// ─── gh#888: the emitted `ir_version` must agree with the key it is filed under ──
//
// The cache key folds the IR schema version the *runtime* expects, never the one
// the compiler actually emitted. With the camdlc↔camdl handshake skipped
// (`CAMDL_SKIP_VERSION_CHECK=1` — every test harness, every ad-hoc worktree
// run) a stale camdlc emitting an older document had its output filed under the
// current schema's key. Every later read of that model then hard-errored on the
// version mismatch instead of missing the cache, and only a manual `rm` cleared
// it. Both halves are pinned below: the mismatched document is never written,
// and a pre-existing poisoned entry recompiles rather than erroring forever.

/// The IR schema version this checkout declares — the version `camdl` expects a
/// document to carry. Read from `ir/VERSION`, the same file both toolchains bake
/// in, so the test never hardcodes a number that a bump would falsify.
fn expected_ir_version() -> String {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let v = Path::new(&manifest).join("../../../ir/VERSION");
    std::fs::read_to_string(v).expect("ir/VERSION must be readable").trim().to_string()
}

/// A camdlc wrapper that stands in for a stale compiler: it compiles for real
/// (so the document is otherwise valid, and `--emit-deps` still lands) but
/// rewrites the envelope's `ir_version` to `fake` on the way out. Compiles are
/// counted as in `counting_shim`; the `--camdl-version` probe passes through
/// untouched.
fn stale_version_shim(dir: &Path, real: &Path, counter: &Path, fake: &str) -> PathBuf {
    let shim = dir.join("camdlc");
    std::fs::write(&shim, format!(
        "#!/bin/sh\n\
         case \"$1\" in\n  \
           --camdl-version) exec '{real}' \"$@\" ;;\n\
         esac\n\
         echo x >> '{counter}'\n\
         out=$('{real}' \"$@\") || exit $?\n\
         printf '%s' \"$out\" | sed '1s/\"ir_version\": *\"[^\"]*\"/\"ir_version\":\"{fake}\"/'\n",
        real = real.display(), counter = counter.display(), fake = fake)).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut p = std::fs::metadata(&shim).unwrap().permissions();
        p.set_mode(0o755);
        std::fs::set_permissions(&shim, p).unwrap();
    }
    dir.to_path_buf()
}

/// Like `run_simulate`, but hands back the process output instead of asserting
/// success — the gh#888 tests are about what happens when a compile is refused.
fn try_simulate(bin: &Path, shim_dir: &Path, model: &Path, cache_dir: &Path, out: &Path)
    -> std::process::Output
{
    let old_path = std::env::var("PATH").unwrap_or_default();
    Command::new(bin)
        .args([
            "simulate", model.to_str().unwrap(),
            "--backend", "chain_binomial", "--seed", "1",
            "--param", "beta=0.3", "--param", "gamma=0.1", "--param", "N0=1000",
            "--output-dir", out.to_str().unwrap(), "--progress", "none",
        ])
        .env("PATH", format!("{}:{}", shim_dir.display(), old_path))
        .env("CAMDL_IR_CACHE_DIR", cache_dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().expect("spawn")
}

/// Every published cache entry, by path (`<key>.ir.json`).
fn cached_entries(cache_dir: &Path) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(cache_dir) else { return Vec::new(); };
    rd.filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().ends_with(".ir.json")))
        .collect()
}

/// The `ir_version` an IR document on disk declares.
fn declared_version_of(entry: &Path) -> String {
    let json = std::fs::read_to_string(entry).unwrap();
    let key = "\"ir_version\"";
    let i = json.find(key).expect("an IR document must declare ir_version");
    let rest = &json[i + key.len()..];
    let colon = rest.find(':').unwrap();
    let q1 = rest[colon..].find('"').unwrap() + colon + 1;
    let q2 = rest[q1..].find('"').unwrap() + q1;
    rest[q1..q2].to_string()
}

/// Rewrite the `ir_version` an IR document on disk declares, leaving the rest of
/// the document (and its `.deps` sidecar) untouched — the shape of a poisoned
/// entry as observed on 2026-09-09.
fn poison_entry(entry: &Path, fake: &str) {
    let json = std::fs::read_to_string(entry).unwrap();
    let was = declared_version_of(entry);
    let patched = json.replacen(
        &format!("\"ir_version\":\"{was}\""),
        &format!("\"ir_version\":\"{fake}\""),
        1);
    assert_ne!(patched, json, "the poisoning rewrite must actually change the document");
    std::fs::write(entry, patched).unwrap();
}

/// gh#888, half one: a compiler that emits a document at the wrong schema
/// version must be refused — and must leave no cache entry behind. Publishing it
/// under the current key is what poisons the cache: the entry then says one
/// version by its key and another by its content, so every later read
/// hard-errors instead of missing.
#[test]
fn stale_compiler_output_is_refused_and_never_cached() {
    let Some((bin, real)) = skip_if_unbuilt() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("sir.camdl");
    std::fs::write(&model, SIR).unwrap();
    let counter = tmp.path().join("compiles.log");
    let shim = stale_version_shim(tmp.path(), &real, &counter, "0.39");
    let cache = tmp.path().join("ircache");

    let out = try_simulate(&bin, &shim, &model, &cache, &tmp.path().join("o1"));
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();

    let entries = cached_entries(&cache);
    assert!(entries.is_empty(),
        "a document at the wrong ir_version must never be published to the cache; \
         found {entries:?}");
    assert!(!out.status.success(),
        "a document at the wrong ir_version must not run. stderr:\n{stderr}");

    let expected = expected_ir_version();
    assert!(stderr.contains("0.39"),
        "the error must name the version the compiler emitted. stderr:\n{stderr}");
    assert!(stderr.contains(&expected),
        "the error must name the version this camdl expects ({expected}). stderr:\n{stderr}");
    assert!(stderr.contains("camdlc"),
        "the error must point at the compiler as the likely cause. stderr:\n{stderr}");
}

/// gh#888, half two: an entry already on disk whose declared `ir_version`
/// disagrees with the key it is filed under must read as a cache miss —
/// recompiled and replaced — not as a hard error. Before the fix this state was
/// terminal: every run of that model failed until someone deleted the file by
/// hand.
#[test]
fn a_poisoned_cache_entry_recompiles_instead_of_erroring_forever() {
    let Some((bin, real)) = skip_if_unbuilt() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("sir.camdl");
    std::fs::write(&model, SIR).unwrap();
    let counter = tmp.path().join("compiles.log");
    let shim = counting_shim(tmp.path(), &real, &counter);
    let cache = tmp.path().join("ircache");

    // A healthy entry, published by a matching compiler.
    run_simulate(&bin, &shim, &model, &cache, &tmp.path().join("o1"), false);
    assert_eq!(compiles(&counter), 1, "first run compiles once (cache miss)");
    let entries = cached_entries(&cache);
    assert_eq!(entries.len(), 1, "one model, one entry: {entries:?}");
    let entry = entries[0].clone();
    let expected = expected_ir_version();
    assert_eq!(declared_version_of(&entry), expected);

    // Poison it exactly as a stale compiler would have: the key still says the
    // current schema, the content now says an older one. The sidecar is left
    // alone, so read()-freshness still passes — the version is the only defect.
    poison_entry(&entry, "0.39");

    let out = try_simulate(&bin, &shim, &model, &cache, &tmp.path().join("o2"));
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(out.status.success(),
        "a version-mismatched entry must be treated as a miss and recompiled, \
         not surfaced as an error. stderr:\n{stderr}");
    assert_eq!(compiles(&counter), 2,
        "the poisoned entry must not be served: the run recompiles");
    assert_eq!(declared_version_of(&entry), expected,
        "the recompile republishes the entry, healing it in place");

    // ...and the healed entry is a plain cache hit thereafter.
    run_simulate(&bin, &shim, &model, &cache, &tmp.path().join("o3"), false);
    assert_eq!(compiles(&counter), 2, "the healed entry is served like any other");
}
