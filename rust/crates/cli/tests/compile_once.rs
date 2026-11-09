//! Regression: a multi-cell `camdl simulate` compiles the `.camdl` source
//! EXACTLY ONCE, not once per cell (replicate / seed / draw / scenario).
//!
//! Before the compile-once hoist, `simulate --replicates 3` routed every
//! engine cell through `util::run_simulation` → `resolve_ir_path`, so camdlc
//! was spawned four times (one validation/CAS load + one per replicate). On a
//! TTY the repeated compile spinners stomped each other; on a large stratified
//! model it was an N× compile cost. The fix resolves `.camdl` → IR once up
//! front and threads the compiled path into the job + every cell.
//!
//! The test wraps the real camdlc in a counting shim placed ahead on PATH and
//! asserts the wrapper is invoked once for a three-replicate run. A lone run is
//! checked for the same one-compile property.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

/// The real camdlc built by the OCaml frontend (`make build-ocaml`).
fn real_camdlc() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest)
        .join("../../../ocaml/_build/default/bin/camdlc.exe")
}

fn skip_if_unbuilt() -> Option<(PathBuf, PathBuf)> {
    let bin = binary();
    let cc = real_camdlc();
    if !bin.exists() {
        eprintln!("skipping: camdl binary not built at {}", bin.display());
        return None;
    }
    if !cc.exists() {
        eprintln!("skipping: camdlc not built at {} (run `make build-ocaml`)",
            cc.display());
        return None;
    }
    Some((bin, cc))
}

/// A minimal SIR with parameter defaults so a bare `simulate` resolves every
/// value without `--param` or a scenario.
fn write_sir(path: &Path) {
    let src = r#"
time_unit = 'days

compartments { S, I, R }

let N = S + I + R

parameters {
  beta  : rate  in [0.001, 2.0]
  gamma : rate  in [0.001, 1.0]
  N0    : count in [100, 100000]
  I0    : count in [1, 1000]
}

transitions {
  infection : S --> I  @ beta * S * (I / N)
  recovery  : I --> R  @ gamma * I
}

init {
  S = N0 - I0
  I = I0
}

simulate { from = 0 'days  to = 20 'days }
"#;
    std::fs::write(path, src).unwrap();
}

/// Build a counting wrapper for camdlc: each *compile* invocation appends a
/// line to `counter_file`, then execs the real camdlc with all args. A bare
/// `--camdl-version` probe (the existence check `find_camdlc` runs once per
/// process) is NOT counted — only real model compiles are. Returns the dir
/// containing the wrapper (to prepend to PATH).
fn counting_camdlc_shim(dir: &Path, real: &Path, counter_file: &Path) -> PathBuf {
    let shim = dir.join("camdlc");
    let script = format!(
        "#!/bin/sh\n\
         case \"$1\" in\n\
           --camdl-version) ;;\n\
           *) echo invoked >> '{counter}' ;;\n\
         esac\n\
         exec '{real}' \"$@\"\n",
        counter = counter_file.display(),
        real = real.display(),
    );
    std::fs::write(&shim, script).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&shim).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&shim, perms).unwrap();
    }
    dir.to_path_buf()
}

fn count(counter_file: &Path) -> usize {
    std::fs::read_to_string(counter_file)
        .map(|s| s.lines().filter(|l| !l.trim().is_empty()).count())
        .unwrap_or(0)
}

/// Run `simulate <model> --replicates <n>` with the counting shim on PATH and
/// return the number of camdlc invocations.
fn run_and_count_compiles(bin: &Path, real: &Path, replicates: usize) -> (usize, std::process::Output) {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("sir.camdl");
    write_sir(&model);
    let counter = tmp.path().join("compiles.log");
    let shim_dir = counting_camdlc_shim(tmp.path(), real, &counter);
    let out_dir = tmp.path().join("store");

    let old_path = std::env::var("PATH").unwrap_or_default();
    let new_path = format!("{}:{}", shim_dir.display(), old_path);

    let out = Command::new(bin).env_remove("CLICOLOR_FORCE")
        .args([
            "simulate",
            &model.to_string_lossy(),
            // Concrete param values (the DSL has no inline default form; values
            // come from --param / a scenario). These keep the run self-contained.
            "--param", "beta=0.3",
            "--param", "gamma=0.1",
            "--param", "N0=1000",
            "--param", "I0=10",
            "--replicates",
            &replicates.to_string(),
            "--progress",
            "none",
            "--output-dir",
            &out_dir.to_string_lossy(),
            "-o",
            &tmp.path().join("traj.tsv").to_string_lossy(),
        ])
        // Skip the camdlc↔camdl version handshake: the shim wraps a freshly
        // built camdlc whose git hash need not match the test binary's, and
        // the probe would otherwise spawn camdlc an extra (uncounted) time.
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        // Disable the compiled-IR cache: this test measures the WITHIN-process
        // compile-once hoist, so each resolve_ir_path call must actually
        // compile. With the cache on, a regressed (per-cell) hoist would hit
        // the cache after the first compile and still count 1 — masking the
        // regression. The cache has its own test (ir_cache.rs).
        .env("CAMDL_NO_IR_CACHE", "1")
        .env("PATH", &new_path)
        .output()
        .expect("camdl simulate must spawn");

    (count(&counter), out)
}

/// A three-replicate ensemble must compile the model exactly once — not once
/// per replicate. This is the core regression guard.
#[test]
fn multicell_simulate_compiles_camdl_once() {
    let Some((bin, real)) = skip_if_unbuilt() else { return; };
    let (compiles, out) = run_and_count_compiles(&bin, &real, 3);
    assert!(
        out.status.success(),
        "simulate --replicates 3 must succeed. stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        compiles, 1,
        "a 3-replicate `simulate` must compile the .camdl exactly once \
         (compile-once hoist); got {} camdlc invocations",
        compiles
    );
}

/// A lone run shares the same compile-once path.
#[test]
fn single_cell_simulate_compiles_camdl_once() {
    let Some((bin, real)) = skip_if_unbuilt() else { return; };
    let (compiles, out) = run_and_count_compiles(&bin, &real, 1);
    assert!(
        out.status.success(),
        "single simulate must succeed. stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        compiles, 1,
        "a single `simulate` must compile the .camdl exactly once; got {}",
        compiles
    );
}
