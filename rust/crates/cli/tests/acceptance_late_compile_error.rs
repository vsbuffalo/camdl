//! Acceptance test (gh#181): a POST-EXPANSION compile error must exit cleanly.
//!
//! Before the fix, `Compiler.compile` returned `(Ir.model, string) result` but
//! *raised* `Compile_error` on late-phase errors (Validate E5xx, autodiff
//! E600). `camdlc.ml`'s top level has no exception handler, so the raise
//! escaped: the user saw their rendered diagnostic followed by
//! `Fatal error: exception Diagnostics.Compile_error(...)` and the process
//! exited 2. Front-end (lex/parse/expand) errors took the clean `Error`-return
//! path and exited 1 — so the exit code depended on *which phase* rejected the
//! model. Repro at the time: `camdlc dangling.camdl` → exit 2 + Fatal-error
//! line.
//!
//! The fix makes `compile` non-raising: late errors return `Error (rendered)`
//! exactly like front-end errors, so `camdlc` exits 1 with no Fatal-error line.
//! The OCaml unit guard is `test_compile_outcome_late_error_is_value_not_raise`
//! in `ocaml/test/test_compiler.ml`; this pins the same property at the binary
//! level, which no in-process OCaml test can (it's about the CLI's
//! uncaught-exception behaviour).

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn skip_if_missing_binary() -> PathBuf {
    let bin = binary();
    assert!(
        bin.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test`",
        bin.display()
    );
    bin
}

/// Non-stratified SEIR whose observation projects a transition that does not
/// exist (`infektion`, a typo for `infection`). A bare unknown name in
/// `incidence(...)` survives expansion as a dangling `CumulativeFlow` and is
/// caught post-expansion by Validate as E507 — the late path that used to
/// raise.
fn write_dangling_incidence_model(path: &Path) {
    let src = r#"
time_unit = 'days

compartments { S, E, I, R }

let N = S + E + I + R

parameters {
  beta  : rate        in [0.001, 0.5]
  sigma : rate        in [0.01,  1.0]
  gamma : rate        in [0.01,  1.0]
  rho   : probability in [0.0,   1.0]
  k     : real        in [0.1,  100.0]
}

transitions {
  infection   : S --> E  @ beta * S * I / N
  progression : E --> I  @ sigma * E
  recovery    : I --> R  @ gamma * I
}

observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    projected     = incidence(infektion)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}

init { S = 100  I = 1 }

simulate { from = 0 'days  to = 10 'days }
"#;
    std::fs::write(path, src).unwrap();
}

#[test]
fn late_compile_error_exits_clean_no_fatal_trace() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("dangling.camdl");
    write_dangling_incidence_model(&model);

    // `camdl compile` delegates verbatim to `camdlc`, passing the exit code
    // through. CAMDL_SKIP_VERSION_CHECK so a stale ~/.local/bin camdlc can't
    // turn this into a version-mismatch failure instead of the E507 we want.
    let out = Command::new(&bin)
        .args(["compile", &model.to_string_lossy()])
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl compile");

    let stderr = String::from_utf8_lossy(&out.stderr);

    // It must still REJECT the model (the dangling reference is a real error)...
    assert!(
        !out.status.success(),
        "a dangling observation reference must be rejected. stderr:\n{stderr}"
    );
    // ...with the E507 diagnostic...
    assert!(
        stderr.contains("E507"),
        "expected the E507 diagnostic for the dangling transition, got:\n{stderr}"
    );
    // ...and crucially: a CLEAN exit 1, not the exit-2 uncaught-exception path,
    // and no OCaml Fatal-error trace tacked onto the diagnostic.
    assert!(
        !stderr.contains("Fatal error"),
        "the diagnostic must not be followed by an OCaml Fatal-error trace \
         (the gh#181 regression). stderr:\n{stderr}"
    );
    assert_eq!(
        out.status.code(),
        Some(1),
        "a post-expansion compile error must exit 1 (clean), not 2 \
         (uncaught Compile_error). stderr:\n{stderr}"
    );
}
