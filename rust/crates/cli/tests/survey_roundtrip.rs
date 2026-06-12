//! gh#147 (M3.3) — end-to-end round-trip for the content-addressed
//! `camdl survey` writer and its `list` / `show` / `cat` readers.
//!
//! The `survey_top_k_*` tests feed a *hand-written* survey fixture into
//! a fit; none of them exercise the real `camdl survey` writer. This
//! test runs the actual command and asserts the leaf it produces is
//! discoverable and readable through every browse path:
//!
//!   1. `camdl survey … --eval simulate` writes a `surveys/…` leaf with
//!      a `runid::RunRecord` `run.json` + a `landscape.tsv`.
//!   2. `camdl list --kind survey --format json` surfaces the leaf's
//!      `run_id` and recorded `inputs` (n_points, estimated, eval).
//!   3. `camdl show <run_id[:8]>` resolves the hash prefix and prints
//!      `kind: survey` plus the factored levels.
//!   4. `camdl cat <run_id[:8]>` emits the `landscape.tsv` body.
//!
//! `--eval simulate` keeps the run deterministic and sub-second (no PF).
//! Skipped when the release binary or camdlc isn't present, mirroring
//! the gate in `survey_top_k_pmmh.rs`.

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

fn camdlc_bin() -> Option<PathBuf> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    if p.exists() { Some(p) } else { None }
}

struct Tmp(PathBuf);
impl Tmp { fn path(&self) -> &Path { &self.0 } }
impl Drop for Tmp { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
fn tempdir(tag: &str) -> Tmp {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let base = std::env::temp_dir().join(format!(
        "camdl_survey_roundtrip_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

/// A tiny deterministic SIR model + dataset (mirrors the fixture in
/// `survey_top_k_pmmh.rs`, minus the PF-specific bits).
fn write_fixture(dir: &Path) -> (PathBuf, PathBuf) {
    let camdlc = camdlc_bin().expect("camdlc.exe present");
    let src = r#"
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
simulate { from = 0 'days  to = 6 'days }
"#;
    let model_path = dir.join("sir.camdl");
    std::fs::write(&model_path, src).unwrap();
    let ir_path = dir.join("sir.ir.json");
    let out = Command::new(&camdlc).arg(&model_path).output().unwrap();
    assert!(out.status.success(),
        "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(&ir_path, &out.stdout).unwrap();

    let data_path = dir.join("cases.tsv");
    std::fs::write(&data_path,
        "time\tcases\n1\t2\n2\t4\n3\t8\n4\t6\n5\t4\n6\t2\n").unwrap();

    (ir_path, data_path)
}

/// The single `surveys/…` leaf directory (the one holding `landscape.tsv`).
fn find_survey_leaf(root: &Path) -> PathBuf {
    let surveys = root.join("surveys");
    let mut stack = vec![surveys];
    while let Some(dir) = stack.pop() {
        if dir.join("landscape.tsv").is_file() && dir.join("run.json").is_file() {
            return dir;
        }
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() { stack.push(p); }
            }
        }
    }
    panic!("no survey leaf with landscape.tsv + run.json under {}", root.display());
}

#[test]
fn survey_write_then_list_show_cat_roundtrip() {
    // `camdl_bin()` is fail-loud (gh#105); `write_fixture` below already
    // `.expect()`s camdlc.exe, so a missing OCaml compiler also surfaces
    // loudly rather than skipping.
    let bin = camdl_bin();

    let tmp = tempdir("rt");
    let (ir, data) = write_fixture(tmp.path());
    let root = tmp.path().join("results");

    // ── 1. Write: run the real `camdl survey` command. ──────────────
    let out = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "survey", &ir.to_string_lossy(),
            "--data", &data.to_string_lossy(),
            "--estimate", "beta=0.001:5.0",
            "--estimate", "gamma=0.01:1.0",
            "--fixed", "N0=1000",
            "--eval", "simulate",
            "--n-points", "8",
            "--seed", "1",
            "--output", &root.to_string_lossy(),
        ])
        .output().expect("spawn camdl survey");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(),
        "camdl survey must exit 0.\nstderr:\n{}", stderr);

    // The leaf exists with both artifacts.
    let leaf = find_survey_leaf(&root);
    assert!(leaf.join("landscape.tsv").is_file(), "landscape.tsv missing");
    assert!(leaf.join("run.json").is_file(), "run.json missing");

    // The run.json is a parseable RunRecord of kind Survey, and carries
    // the cross-check provenance the survey_top_k consumer reads back.
    let rec_bytes = std::fs::read(leaf.join("run.json")).unwrap();
    let record: runid::RunRecord = serde_json::from_slice(&rec_bytes)
        .expect("run.json must deserialize as a RunRecord");
    assert_eq!(record.kind, runid::ArtifactKind::Survey,
        "leaf run.json kind must be Survey");
    let run_id = record.run_id.to_hex();
    let inputs = record.inputs.as_object().expect("inputs object");
    assert_eq!(inputs.get("n_points").and_then(|v| v.as_u64()), Some(8),
        "inputs.n_points must round-trip");
    assert_eq!(inputs.get("eval_method").and_then(|v| v.as_str()), Some("simulate"),
        "inputs.eval_method must round-trip");
    assert!(inputs.get("model_hash").and_then(|v| v.as_str()).is_some(),
        "inputs.model_hash (cross-check provenance) must be present");
    assert!(inputs.get("data_hashes").and_then(|v| v.as_object()).is_some_and(|m| m.contains_key("cases")),
        "inputs.data_hashes must record the `cases` stream");

    let prefix = &run_id[..8];

    // ── 2. list --kind survey --format json: leaf is discoverable. ──
    // `root` is resolved from CAMDL_OUTPUT_DIR (positional default) so the
    // same env works uniformly across list / show / cat.
    let out = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_OUTPUT_DIR", &*root.to_string_lossy())
        .args(["list", "--kind", "survey", "--format", "json"])
        .output().expect("spawn camdl list");
    assert!(out.status.success(), "camdl list must exit 0.\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr));
    let list_json = String::from_utf8_lossy(&out.stdout);
    assert!(list_json.contains(&run_id),
        "list --kind survey must surface the leaf's run_id {}.\ngot:\n{}",
        run_id, list_json);

    // ── 3. show <prefix>: resolves the hash prefix, prints kind survey. ─
    let out = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_OUTPUT_DIR", &*root.to_string_lossy())
        .args(["show", prefix])
        .output().expect("spawn camdl show");
    assert!(out.status.success(),
        "camdl show {} must exit 0.\nstderr:\n{}",
        prefix, String::from_utf8_lossy(&out.stderr));
    let show_out = String::from_utf8_lossy(&out.stdout);
    assert!(show_out.contains("survey"),
        "show must label the leaf as a survey:\n{}", show_out);
    assert!(show_out.contains(&run_id),
        "show must print the full run_id:\n{}", show_out);

    // ── 4. cat <prefix>: emits the landscape.tsv body. ──────────────
    let out = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_OUTPUT_DIR", &*root.to_string_lossy())
        .args(["cat", prefix])
        .output().expect("spawn camdl cat");
    assert!(out.status.success(),
        "camdl cat {} must exit 0.\nstderr:\n{}",
        prefix, String::from_utf8_lossy(&out.stderr));
    let cat_out = String::from_utf8_lossy(&out.stdout);
    // Landscape header carries the estimated-param columns + loglik.
    assert!(cat_out.contains("beta") && cat_out.contains("gamma")
            && cat_out.contains("loglik"),
        "cat must emit the landscape.tsv body (header beta/gamma/loglik):\n{}",
        cat_out);
    // Sanity: the cat output equals the leaf's landscape.tsv on disk.
    let on_disk = std::fs::read_to_string(leaf.join("landscape.tsv")).unwrap();
    assert_eq!(cat_out, on_disk,
        "cat must reproduce the leaf's landscape.tsv byte-for-byte");
}
