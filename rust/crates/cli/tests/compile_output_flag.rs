//! Regression test for `camdl compile -o FILE` / `--output FILE`
//! (closes GH #23). Pre-fix, `camdlc` had no output flag — IR JSON
//! went to stdout only. Fixed in `ocaml/bin/camdlc.ml`.
//!
//! The test compiles a tiny .camdl source three ways and asserts:
//!
//! 1. Stdout path (no flag) still works and produces non-empty JSON.
//! 2. `-o FILE` writes the same bytes the stdout path would have
//!    written.
//! 3. `--output FILE` (long form) produces byte-identical output to
//!    `-o FILE`.
//!
//! Silent-skip if the binaries can't be located (e.g. running the
//! test outside the repo's build layout). Same convention as
//! `fit_experiment_management.rs`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set under cargo test");
    let p = Path::new(&manifest).join("../../target/release/camdl");
    assert!(
        p.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test` (gh#105)",
        p.display()
    );
    p
}

fn tempdir(tag: &str) -> PathBuf {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let p = std::env::temp_dir().join(format!(
        "camdl_compile_output_{}_{}_{}",
        tag, std::process::id(), ns));
    std::fs::create_dir_all(&p).unwrap();
    p
}

/// Tiny SIR source. The same shape `fit_experiment_management::build_fixture`
/// uses, copy-inlined here so this test file stays self-contained.
const TINY_SIR: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.001, 5.0]
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 10000]
}
transitions {
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected  = prevalence(I)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 999  I = 1 }
simulate { from = 0 'days  to = 10 'days }
"#;

#[test]
fn camdl_compile_dash_o_writes_to_file_byte_identical_to_stdout() {
    let camdl = camdl_bin();

    let tmp = tempdir("dash_o");
    let model = tmp.join("sir.camdl");
    std::fs::write(&model, TINY_SIR).unwrap();

    // (1) Stdout path: capture command stdout.
    let stdout_out = Command::new(&camdl)
        .arg("compile")
        .arg(&model)
        .output()
        .expect("camdl compile must invoke");
    assert!(
        stdout_out.status.success(),
        "stdout path failed: stderr={}",
        String::from_utf8_lossy(&stdout_out.stderr));
    assert!(
        !stdout_out.stdout.is_empty(),
        "stdout path produced empty output");
    // Sanity: this looks like JSON.
    assert!(
        stdout_out.stdout.starts_with(b"{"),
        "stdout output should start with `{{`; got first 40 bytes: {:?}",
        &stdout_out.stdout[..stdout_out.stdout.len().min(40)]);

    // (2) -o FILE: writes IR JSON to the file.
    let dash_o_path = tmp.join("dash_o.ir.json");
    let dash_o_status = Command::new(&camdl)
        .arg("compile").arg(&model)
        .arg("-o").arg(&dash_o_path)
        .status()
        .expect("camdl compile -o must invoke");
    assert!(dash_o_status.success(),
        "-o FILE invocation failed (exit {:?})", dash_o_status.code());
    let dash_o_bytes = std::fs::read(&dash_o_path)
        .expect("camdl compile -o must produce the named file");
    assert!(
        !dash_o_bytes.is_empty(),
        "-o produced an empty file");

    // (3) -o output must be byte-identical to what stdout produced.
    // (camdlc emits the same JSON in both modes, including the
    //  trailing newline.)
    assert_eq!(
        dash_o_bytes, stdout_out.stdout,
        "-o FILE output must be byte-identical to stdout output");

    // (4) --output FILE long form: same bytes as -o.
    let long_path = tmp.join("long.ir.json");
    let long_status = Command::new(&camdl)
        .arg("compile").arg(&model)
        .arg("--output").arg(&long_path)
        .status()
        .expect("camdl compile --output must invoke");
    assert!(long_status.success(),
        "--output FILE invocation failed (exit {:?})", long_status.code());
    let long_bytes = std::fs::read(&long_path)
        .expect("camdl compile --output must produce the named file");
    assert_eq!(
        long_bytes, dash_o_bytes,
        "--output FILE must produce byte-identical output to -o FILE");

    // Cleanup.
    let _ = std::fs::remove_dir_all(&tmp);
}
