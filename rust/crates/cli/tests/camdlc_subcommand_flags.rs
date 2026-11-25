//! gh#1216: the flags of `camdlc check`, `camdlc inspect` and `camdlc doctest`.
//!
//! The flag parsing lives in `ocaml/bin/camdlc.ml`, the compiler's entry
//! point, which no OCaml test links, so these run the built compiler:
//!
//! - `inspect --help`, `doctest --help` and `render --help` print help and
//!   exit 0, as the error for an unknown flag and the Makefile tell the
//!   reader to run them;
//! - `inspect --ir`, which no report read, is refused as an unknown flag;
//! - `--count` with `--transitions` prints counts, and with `--transition NAME`
//!   counts only the declarations that name matches.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn repo_root() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../..").canonicalize().unwrap()
}

/// The real camdlc built by the OCaml frontend (`make build-ocaml`), run
/// directly so the runtime/compiler version handshake is not in play.
fn camdlc() -> Option<PathBuf> {
    let cc = repo_root().join("ocaml/_build/default/bin/camdlc.exe");
    if cc.exists() {
        Some(cc)
    } else {
        eprintln!("skipping: camdlc not built at {} (run `make build-ocaml`)", cc.display());
        None
    }
}

const SIR: &str = "time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate in [0.001, 2.0]
  gamma : rate in [0.001, 1.0]
}
transitions {
  infection : S --> I @ beta * S * I / (S + I + R)
  recovery  : I --> R @ gamma * I
}
init {
  S = 990
  I = 10
}
simulate { from = 0 'days
  to = 50 'days }
";

fn run(cc: &Path, args: &[&str]) -> Output {
    Command::new(cc)
        .args(args)
        .env("NO_COLOR", "1")
        .output()
        .expect("spawn camdlc")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn all_output(o: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr))
}

fn write(dir: &Path, name: &str, text: &str) -> String {
    let p = dir.join(name);
    std::fs::write(&p, text).unwrap();
    p.to_str().unwrap().to_string()
}

#[test]
fn subcommands_take_help() {
    let Some(cc) = camdlc() else { return };
    let o = run(&cc, &["inspect", "--help"]);
    assert!(o.status.success(), "inspect --help: {}", all_output(&o));
    assert!(stdout(&o).contains("--transitions"), "inspect --help lists no flags: {}", stdout(&o));

    let o = run(&cc, &["doctest", "--help"]);
    assert!(o.status.success(), "doctest --help: {}", all_output(&o));
    assert!(stdout(&o).contains("--gate"), "doctest --help lists no flags: {}", stdout(&o));

    let o = run(&cc, &["render", "--help"]);
    assert!(o.status.success(), "render --help: {}", all_output(&o));
    assert!(stdout(&o).contains("--format"), "render --help lists no flags: {}", stdout(&o));
}

#[test]
fn inspect_refuses_ir_flag() {
    let Some(cc) = camdlc() else { return };
    let dir = tempfile::tempdir().unwrap();
    let model = write(dir.path(), "sir.camdl", SIR);
    let o = run(&cc, &["inspect", &model, "--ir"]);
    assert!(!o.status.success(), "--ir accepted: {}", all_output(&o));
    assert!(all_output(&o).contains("unknown flag '--ir'"), "{}", all_output(&o));
}

#[test]
fn count_wins_and_is_narrowed_by_the_name() {
    let Some(cc) = camdlc() else { return };
    let dir = tempfile::tempdir().unwrap();
    let model = write(dir.path(), "sir.camdl", SIR);

    let o = run(&cc, &["inspect", &model, "--transitions", "--count"]);
    assert!(o.status.success(), "{}", all_output(&o));
    let out = stdout(&o);
    assert!(out.contains("all combos"), "--transitions --count printed no counts:\n{out}");
    assert!(!out.contains('@'), "--transitions --count printed rates:\n{out}");

    let o = run(&cc, &["inspect", &model, "--transition", "infection", "--count"]);
    assert!(o.status.success(), "{}", all_output(&o));
    let out = stdout(&o);
    assert!(out.contains("infection"), "the named transition is missing:\n{out}");
    assert!(!out.contains("recovery"), "--transition infection --count counted recovery:\n{out}");
}
