//! ir/VERSION 0.33: a forcing declared `data = "path"` records WHICH file its
//! knots came from and WHAT was in it.
//!
//! The knots are read by `camdlc` and inlined into the IR, so the runtime
//! never opens the file — which is why, before this, a completed fit could not
//! name the one input most likely to change underneath it.
//!
//! Three things are pinned here, and the first is the cross-language contract:
//!
//! 1. **The OCaml compiler's SHA-256 agrees with Rust's `sha2`.** `camdlc`
//!    hashes the file with a hand-written SHA-256 (`ocaml/lib/compiler/sha256.ml`
//!    — the opam switch has no digest library); every Rust-side provenance
//!    record uses `sha2::Sha256`. Two implementations of one constant is
//!    exactly the case `.claude/rules/ir-schema.md` says to pin with an
//!    equivalence test, so this reads the committed golden IR and re-hashes
//!    the same file here.
//!
//! 2. **A live compile records it**, with the path AS WRITTEN (relative,
//!    including its directory — not a basename and not the resolved absolute
//!    path, which would bake one machine's layout into the IR).
//!
//! 3. **The hash tracks the bytes, and the knots track the values.** Editing
//!    only a comment line in the data file moves the recorded hash and leaves
//!    the inlined knots identical — which is the whole reason the hash is
//!    provenance and not run identity (`runid::ir_hash`:
//!    `ir_forcing_data_source_excluded_from_hash`).

use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

fn repo_root() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../..").canonicalize().unwrap()
}

/// The real camdlc built by the OCaml frontend (`make build-ocaml`). Invoked
/// directly, not through `camdl`, so the runtime/compiler git-hash handshake
/// (CLAUDE.md, "camdlc version mismatch") is not in play at all.
fn camdlc() -> Option<PathBuf> {
    let cc = repo_root().join("ocaml/_build/default/bin/camdlc.exe");
    if cc.exists() {
        Some(cc)
    } else {
        eprintln!("skipping: camdlc not built at {} (run `make build-ocaml`)", cc.display());
        None
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn load(path: &Path) -> ir::Model {
    let contents = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    ir::from_str(&contents)
        .unwrap_or_else(|e| panic!("failed to deserialise {}: {e}", path.display()))
}

fn forcing<'a>(m: &'a ir::Model, name: &str) -> &'a ir::time_func::TimeFunction {
    m.time_functions
        .iter()
        .find(|tf| tf.name == name)
        .unwrap_or_else(|| panic!("no forcing named {name}"))
}

/// (1) The committed golden is what `camdlc` emitted; re-hash the same file
/// with `sha2` and require the two digests to be equal. If the hand-written
/// SHA-256 ever drifts from the standard, this is where it surfaces.
#[test]
fn ocaml_digest_agrees_with_sha2_over_the_committed_golden() {
    let root = repo_root();
    let model = load(&root.join("ocaml/golden/flu_data_forcing.ir.json"));
    let clim = forcing(&model, "clim");

    let ds = clim
        .data_source
        .as_ref()
        .expect("flu_data_forcing declares `data = \"data/flu_forcing.tsv\"`, so the \
                 compiled IR must name that file");

    assert_eq!(
        ds.path, "data/flu_forcing.tsv",
        "the recorded path must be the string AS WRITTEN in the model — relative, \
         with its directory — so it stays portable across machines and checkouts"
    );

    let bytes = std::fs::read(root.join("ocaml/golden/data/flu_forcing.tsv")).unwrap();
    assert_eq!(
        ds.sha256,
        sha256_hex(&bytes),
        "camdlc's SHA-256 must equal sha2::Sha256 over the same bytes — two \
         implementations of one constant, pinned by this equivalence test"
    );
    assert_eq!(ds.sha256.len(), 64, "the recorded digest is full-length hex");
    assert!(
        ds.sha256.chars().all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
        "the recorded digest is lowercase hex, so `shasum -a 256` output compares \
         directly: {}",
        ds.sha256
    );
}

/// A forcing that reads no file records no provenance — the field appends only
/// when present, which is what keeps every other model's IR byte-identical
/// across this schema bump.
#[test]
fn a_forcing_with_no_data_file_records_nothing() {
    let model = load(&repo_root().join("ocaml/golden/seir_seasonal_patch.ir.json"));
    assert!(
        !model.time_functions.is_empty(),
        "fixture must carry at least one forcing to make this assertion mean something"
    );
    for tf in &model.time_functions {
        assert!(
            tf.data_source.is_none(),
            "forcing '{}' reads no external file, so it must carry no data_source",
            tf.name
        );
    }
}

const MODEL: &str = r#"
time_unit = 'days

compartments { S, I, R }

let N = S + I + R

parameters {
  beta  : rate  in [0.001, 2.0]
  gamma : rate  in [0.001, 1.0]
  N0    : count in [100, 100000]
  I0    : count in [1, 1000]
}

forcing {
  clim : interpolated 'ratio {
    data      = "inputs/clim.tsv"
    time_col  = "t"
    value_col = "force"
    method    = linear
  }
}

transitions {
  infection : S --> I  @ beta * clim(t) * S * (I / N)
  recovery  : I --> R  @ gamma * I
}

init {
  S = N0 - I0
  I = I0
}

simulate {
  from = 0 'days
  to   = 100 'days
}

scenarios {
  baseline {
    set = {
      beta  = 0.3
      gamma = 0.1
      N0    = 1000
      I0    = 10
    }
  }
}
"#;

const DATA_ROWS: &str = "t\tforce\n0\t1.0\n30\t1.4\n60\t0.8\n90\t1.1\n";

/// Compile `MODEL` against a data file whose leading comment block is
/// `header`, returning the compiled model plus the data file's bytes.
fn compile_with_header(cc: &Path, dir: &Path, header: &str) -> (ir::Model, Vec<u8>) {
    std::fs::create_dir_all(dir.join("inputs")).unwrap();
    let data_path = dir.join("inputs/clim.tsv");
    let contents = format!("{header}{DATA_ROWS}");
    std::fs::write(&data_path, &contents).unwrap();
    let model_path = dir.join("m.camdl");
    std::fs::write(&model_path, MODEL).unwrap();
    let out = dir.join("m.ir.json");

    let status = Command::new(cc)
        .arg(&model_path)
        .arg("-o")
        .arg(&out)
        .output()
        .expect("failed to run camdlc");
    assert!(
        status.status.success(),
        "camdlc failed on the fixture model:\n{}",
        String::from_utf8_lossy(&status.stderr)
    );

    (load(&out), contents.into_bytes())
}

/// (2) and (3). A live compile names the file and hashes its bytes; and a byte
/// change that moves no value moves the hash while leaving the knots alone.
#[test]
fn a_live_compile_records_the_file_and_tracks_its_bytes() {
    let Some(cc) = camdlc() else { return };
    let tmp = tempfile::tempdir().unwrap();

    let (plain, plain_bytes) = compile_with_header(&cc, tmp.path(), "");
    let ds = forcing(&plain, "clim")
        .data_source
        .as_ref()
        .expect("a live compile of a `data = \"...\"` forcing must record its source");
    assert_eq!(
        ds.path, "inputs/clim.tsv",
        "the path is recorded as written — relative to the .camdl file, directory \
         included, never resolved to this machine's absolute layout"
    );
    assert!(
        !ds.path.starts_with('/'),
        "an absolute path in the IR would bake one filesystem into the model: {}",
        ds.path
    );
    assert_eq!(ds.sha256, sha256_hex(&plain_bytes));

    // The same data, with a provenance comment block ahead of the header — the
    // repo's own data-step convention (gh#144), which the loader skips. The
    // bytes differ; not one compiled value does.
    let tmp2 = tempfile::tempdir().unwrap();
    let (commented, commented_bytes) = compile_with_header(
        &cc,
        tmp2.path(),
        "# source: https://example.invalid/clim\n# fetched: 2026-08-19\n",
    );
    let ds2 = forcing(&commented, "clim").data_source.as_ref().unwrap();

    assert_eq!(ds2.sha256, sha256_hex(&commented_bytes));
    assert_ne!(
        ds.sha256, ds2.sha256,
        "the recorded hash is over the FILE's bytes, so adding a comment line \
         must move it — that is what makes it useful provenance"
    );
    assert_eq!(
        forcing(&plain, "clim").kind,
        forcing(&commented, "clim").kind,
        "…while the inlined knots are identical, which is precisely why the hash \
         is NOT folded into run identity: folding it would re-key a fit whose \
         model did not change"
    );
}

/// Recorded provenance nobody can read is not provenance. `camdlc inspect
/// --forcings` (reached as `camdl inspect … --forcings`, which forwards
/// verbatim) is the way to ask a model which file each forcing compiled
/// against — one line per forcing, naming the path and the digest's first 8
/// hex digits.
#[test]
fn inspect_forcings_reports_the_file_and_a_hash_prefix() {
    let Some(cc) = camdlc() else { return };
    let root = repo_root();

    let out = Command::new(&cc)
        .arg("inspect")
        .arg(root.join("ocaml/golden/flu_data_forcing.camdl"))
        .arg("--forcings")
        .arg("--no-color")
        .output()
        .expect("failed to run camdlc inspect");
    assert!(
        out.status.success(),
        "camdlc inspect --forcings failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);

    let bytes = std::fs::read(root.join("ocaml/golden/data/flu_forcing.tsv")).unwrap();
    let prefix = &sha256_hex(&bytes)[..8];

    for needle in ["clim", "data/flu_forcing.tsv", prefix] {
        assert!(
            text.contains(needle),
            "`inspect --forcings` must report {needle:?} so a reader can tell WHICH \
             file was compiled in and WHETHER it is the one they think.\nGot:\n{text}"
        );
    }

    // A forcing that reads no file is still listed — the command answers "what
    // forcings does this model have", not only "which ones read a file" — but
    // it must not invent a source for one.
    let out = Command::new(&cc)
        .arg("inspect")
        .arg(root.join("ocaml/golden/seir_seasonal_patch.camdl"))
        .arg("--forcings")
        .arg("--no-color")
        .output()
        .expect("failed to run camdlc inspect");
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("seasonal_urban"), "got:\n{text}");
    assert!(
        !text.contains("sha256"),
        "a forcing that reads no file must show no digest:\n{text}"
    );
}
