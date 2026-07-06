//! Integration tests for the experiment-management foundation.
//!
//! Covers Deliverables A, B, and C from
//! `docs/dev/proposals/2026-04-28-fit-experiment-management.md`:
//!
//! - **A — end-to-end summary walks a real `cmd_fit_run_v2` output.**
//!   Runs the runner, then `camdl fit summary --format json`, and
//!   asserts the output contains a non-empty `stages` array. This is
//!   the structural defence against the v1-layout bug (audit §2.3).
//!
//! - **B — spec/code parity check.** Parses every fenced code block
//!   in `docs/camdl-inference-spec.md` and `docs/inference.md` for
//!   paths shaped `<fit_dir>/...` and asserts each one exists under
//!   the real fit_dir produced by the runner. Fragile-but-loud: if
//!   the spec drifts (introduces a placeholder convention the
//!   parser doesn't understand, or documents a path the runner
//!   doesn't produce), the test fails immediately.
//!
//! - **C — `summary ⊆ table` byte-equality.** Asserts that
//!   `summary_json["table_row"]` is byte-equal to
//!   `table_json["rows"][0]` for the same fit. The two surfaces
//!   share one schema; any field added to one without the other
//!   makes this fail.
//!
//! Both tests shell out to the built `camdl` and `camdlc.exe`
//! binaries; skipped silently when either is absent so the suite
//! stays runnable in rust-only CI and when tests run before a build.

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
    if p.exists() {
        Some(p)
    } else {
        None
    }
}

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR for `cli` = `<workspace>/rust/crates/cli/`.
    // Workspace root (where `docs/` lives) is three levels up.
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR is set during cargo test");
    PathBuf::from(manifest).join("../../..")
}

struct TempDir(PathBuf);
impl TempDir {
    fn path(&self) -> &Path {
        &self.0
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn tempdir(tag: &str) -> TempDir {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let p = std::env::temp_dir().join(format!(
        "camdl_xptmgmt_{}_{}_{}",
        tag,
        std::process::id(),
        ns
    ));
    std::fs::create_dir_all(&p).unwrap();
    TempDir(p)
}

/// Compile a tiny SIR model and emit the IR JSON. Returns
/// (ir_path, data_path).
fn build_fixture(camdlc: &Path, dir: &Path) -> (PathBuf, PathBuf) {
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
simulate { from = 0 'days  to = 10 'days }
"#;
    let model_path = dir.join("sir.camdl");
    std::fs::write(&model_path, src).unwrap();
    let ir_path = dir.join("sir.ir.json");
    let output = Command::new(camdlc).arg(&model_path).output().unwrap();
    assert!(
        output.status.success(),
        "camdlc failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::write(&ir_path, &output.stdout).unwrap();

    // Tiny synthetic data — 10 weekly cases. Doesn't matter for
    // structural tests; we just need the runner to write a stage tree.
    let data_path = dir.join("cases.tsv");
    std::fs::write(
        &data_path,
        "time\tcases\n1\t5\n2\t7\n3\t12\n4\t18\n5\t25\n6\t30\n7\t28\n8\t22\n9\t15\n10\t10\n",
    )
    .unwrap();
    (ir_path, data_path)
}

/// Write a tiny IF2 fit.toml that runs in seconds. 2 chains, 5 iters,
/// 50 particles — enough to populate the v2 stage tree without
/// converging on anything meaningful (Deliverable A is structural,
/// not statistical).
fn write_fit_toml(dir: &Path, ir: &Path, data: &Path, output_dir: &Path) -> PathBuf {
    let fit_toml = dir.join("fit.toml");
    let body = format!(
        r#"
output_dir = "{out}"

[model]
camdl = "{ir}"

[data.observations]
cases = "{data}"

[estimate]
beta  = {{ bounds = [0.01, 5.0], start = 1.0 }}
gamma = {{ bounds = [0.01, 1.0], start = 0.3 }}

[fixed]
N0 = 1000

[stages.scout]
algorithm     = "if2"
backend     = "chain_binomial"
chains     = 2
particles  = 50
iterations = 5
cooling    = 0.7
"#,
        out = output_dir.display(),
        ir = ir.display(),
        data = data.display(),
    );
    std::fs::write(&fit_toml, body).unwrap();
    fit_toml
}

/// Run `camdl fit run <fit_toml>` and return the produced fit_dir
/// (the single child of `<output_dir>/fits/`).
fn exec_fit_run_v2(camdl: &Path, fit_toml: &Path, output_dir: &Path) -> PathBuf {
    let status = Command::new(camdl)
        .arg("fit")
        .arg("run")
        .arg(fit_toml)
        .status()
        .expect("camdl fit run must invoke");
    assert!(status.success(), "camdl fit run failed");
    let fits = output_dir.join("fits");
    let entries: Vec<PathBuf> = std::fs::read_dir(&fits)
        .unwrap_or_else(|_| panic!("no fits/ dir under {}", output_dir.display()))
        .flatten()
        .map(|e| e.path())
        .collect();
    assert_eq!(
        entries.len(),
        1,
        "expected exactly one fit dir under {}, got {:?}",
        fits.display(),
        entries
    );
    entries.into_iter().next().unwrap()
}

fn exec_fit_summary_json(camdl: &Path, fit_dir: &Path) -> serde_json::Value {
    let output = Command::new(camdl)
        .arg("fit")
        .arg("summary")
        .arg(fit_dir)
        .arg("--format")
        .arg("json")
        .arg("--no-color")
        .output()
        .expect("camdl fit summary must invoke");
    assert!(
        output.status.success(),
        "camdl fit summary failed: stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap_or_else(|e| {
        panic!(
            "camdl fit summary --format json did not emit valid JSON: {}\nstdout={}",
            e,
            String::from_utf8_lossy(&output.stdout)
        )
    })
}

/// gh#147 (M3.2): the fit-level (FitDigest) hash for a CAS fit, read from any
/// stage leaf's `levels[name=="fit"].hash`. A CAS fit has no fit-wide
/// `run.json`; this is `Run.hash` for the derived fit-level entry — the value
/// `fit table --hash` filters on and `camdl label` resolves. Replaces the
/// pre-M3.2 read of `<fit_dir>/run.json` `.hash`.
fn fit_level_hash(fit_dir: &Path) -> String {
    let mut stack = vec![fit_dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let rj = d.join("run.json");
        if rj.is_file() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(
                &std::fs::read_to_string(&rj).unwrap_or_default(),
            ) {
                if v.get("kind").and_then(|k| k.as_str()) == Some("fit_stage") {
                    if let Some(h) = v["levels"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .find(|l| l["name"].as_str() == Some("fit"))
                        .and_then(|l| l["hash"].as_str())
                    {
                        return h.to_string();
                    }
                }
            }
        }
        if let Ok(es) = std::fs::read_dir(&d) {
            for e in es.flatten() {
                if e.path().is_dir() {
                    stack.push(e.path());
                }
            }
        }
    }
    panic!("no fit_stage leaf with a `fit` level under {}", fit_dir.display());
}

/// The fit-level sidecar label (`<fit_segment>/fit.meta.json`), the
/// authoritative home for a CAS fit's `--label`.
fn sidecar_label(fit_dir: &Path) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fit_dir.join("fit.meta.json")).ok()?,
    )
    .ok()?;
    v.get("label").and_then(|l| l.as_str()).map(String::from)
}

/// gh#147 (M3.2): a stage leaf for `stage_substr` exists under `fit_dir` at the
/// CAS shape `<fit_dir>/<NN-stage>-<h8>/seed_<N>-<h8>/run.json` (kind
/// `fit_stage`). Returns the leaf dir. Replaces the pre-M3.2 hard-coded
/// `<fit_dir>/real/fit_<seed>/<stage>` probe.
fn cas_stage_leaf(fit_dir: &Path, stage_substr: &str) -> Option<PathBuf> {
    let mut stack = vec![fit_dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let rj = d.join("run.json");
        if rj.is_file() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(
                &std::fs::read_to_string(&rj).unwrap_or_default(),
            ) {
                if v.get("kind").and_then(|k| k.as_str()) == Some("fit_stage") {
                    let stage = v["levels"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .find(|l| l["name"].as_str() == Some("stage"))
                        .and_then(|l| l["label"].as_str())
                        .unwrap_or("");
                    if stage.contains(stage_substr) {
                        return Some(d);
                    }
                }
            }
        }
        if let Ok(es) = std::fs::read_dir(&d) {
            for e in es.flatten() {
                if e.path().is_dir() {
                    stack.push(e.path());
                }
            }
        }
    }
    None
}

// ── Deliverable A — end-to-end summary walks v2 output ─────────────

/// The structural defence against the v1-layout bug from audit §2.3
/// (`docs/dev/notes/2026-04-27-fit-experiment-management-audit.md`):
/// `cmd_fit_summary` shipped with a walker hard-coded to
/// `<fit_dir>/<stage>/` while `cmd_fit_run_v2` writes to
/// `<fit_dir>/real/fit_<seed>/<stage>/`. Before this test existed,
/// the failure mode was a silent "(no MLE stages found)" on every
/// real fit dir.
#[test]
fn fit_summary_walks_real_fit_run_v2_output() {
    let camdl = camdl_bin();
    let Some(camdlc) = camdlc_bin() else { return };

    let tmp = tempdir("xpt_a");
    let (ir, data) = build_fixture(&camdlc, tmp.path());
    let output_dir = tmp.path().join("out");
    let fit_toml = write_fit_toml(tmp.path(), &ir, &data, &output_dir);

    let fit_dir = exec_fit_run_v2(&camdl, &fit_toml, &output_dir);

    // Sanity-check the walker found at least one if2 stage somewhere
    // under fit_dir. The summary command is the proxy: if its JSON
    // `stages` is non-empty, the walker landed on a real
    // run.json-bearing v2 stage_dir.
    let json = exec_fit_summary_json(&camdl, &fit_dir);
    let stages = json
        .get("stages")
        .and_then(|s| s.as_array())
        .unwrap_or_else(|| panic!("summary JSON missing `stages` array: {}", json));
    assert!(
        !stages.is_empty(),
        "summary JSON `stages` is empty — walker did not find any stage_dir under {}",
        fit_dir.display()
    );

    // Spot-check: the canonical CAS stage leaf is on disk where the runner
    // promises — `<fit_dir>/<NN-stage>-<h8>/seed_<N>-<h8>/run.json` (gh#147
    // M3.2). Hashes aren't predictable, so assert the *shape*: a `fit_stage`
    // leaf whose `stage` level contains "scout". Locks the CAS layout into the
    // test surface so a future runner change can't silently break the walker.
    let leaf = cas_stage_leaf(&fit_dir, "scout").unwrap_or_else(|| {
        panic!(
            "expected a CAS `scout` stage leaf under {} but none is present",
            fit_dir.display()
        )
    });
    assert!(
        leaf.join("run.json").is_file(),
        "stage leaf {} must hold a run.json",
        leaf.display()
    );
}

// ── Deliverable B — spec/code parity ───────────────────────────────

/// Force the spec and the runner to agree about what gets written
/// where. Once this test exists, a spec layout diagram cannot drift
/// from `cmd_fit_run_v2`'s actual output without breaking CI — the
/// process gap that produced the audit §2.3 bug becomes mechanically
/// detectable.
///
/// Implementation (proposal §B): regex over fenced code blocks for
/// lines matching `<fit_dir>/...`, substitute `<seed>` → `1`, expand
/// brace-lists, drop entries with unresolved placeholders or globs,
/// and assert each resolved path exists under
/// `exec_fit_run_v2()`'s output. Fragile-but-loud is intentional:
/// no markdown AST, no special-casing.
///
/// gh#147 (M3.3-E): the grid cells now land in the content-addressed store
/// (each dataset × fit-seed is its own CAS fit), so the legacy literal paths
/// this test resolves (`real/fit_<seed>/`, `synthetic/ds_NN/fit_<seed>/`) no
/// longer exist. Re-enabling is NOT a layout reframe: the spec docs still
/// describe the legacy tree, and a CAS leaf path is hash-bearing
/// (`fits/<base-h8>/<NN>-mle-<h8>/seed_<S>-<h8>/`), which cannot be matched by
/// the current literal-path-on-disk model. It needs (1) the spec-doc rewrite
/// to the CAS layout and (2) a shape/template verification model (match path
/// *shape* with `<h8>` placeholders, not literal strings). Tracked in gh#159
/// (M4); gh#150 (the grid migration) is closed by M3.3-E.
#[test]
#[ignore = "spec-vs-code parity test: literal-path model is incompatible with \
            hash-bearing CAS paths; needs the spec-doc rewrite + a \
            shape/template verification redesign — M4 (gh#159)"]
fn spec_layout_diagrams_match_fit_run_v2_output() {
    let camdl = camdl_bin();
    let Some(camdlc) = camdlc_bin() else { return };

    let tmp = tempdir("xpt_b");
    let (ir, data) = build_fixture(&camdlc, tmp.path());
    let output_dir = tmp.path().join("out");
    let fit_toml = write_fit_toml(tmp.path(), &ir, &data, &output_dir);

    // Rename the single declared stage to `mle` (already in
    // write_fit_toml). The spec diagrams reference scout / refine /
    // validate, but the runner can be configured to use any stage
    // name. We compare against the spec by running its declared
    // stages — but driving a real scout/refine/validate fit takes
    // far longer than a structural test should. Instead: parse the
    // spec for paths, but only assert on the *layout shape*
    // (everything up through the stage component), not on which
    // exact stage name was declared.
    //
    // Concretely: every documented path under `<fit_dir>/real/...`
    // looks like `<fit_dir>/real/fit_<seed>/<stage>/<...>`. We
    // assert the prefix `<fit_dir>/real/fit_<seed>/` is real on
    // disk. That's enough to catch the v1-layout bug class.
    let fit_dir = exec_fit_run_v2(&camdl, &fit_toml, &output_dir);

    let spec_path = repo_root().join("docs/camdl-inference-spec.md");
    let inference_path = repo_root().join("docs/inference.md");
    let mut documented_paths: Vec<String> = Vec::new();
    for path in [&spec_path, &inference_path] {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read {}: {}", path.display(), e));
        let extracted = parse_layout_diagrams(&text);
        for rel in extracted {
            documented_paths.push(rel);
        }
    }

    assert!(
        !documented_paths.is_empty(),
        "parse_layout_diagrams found zero `<fit_dir>/...` paths in either spec doc — \
         either the parser is wrong or the spec has stopped using the canonical \
         `<fit_dir>` placeholder"
    );

    // Whittle down to paths whose *prefix* we can assert against the
    // real fit_dir. We can't reliably assert the leaf
    // (stage names differ between docs and this fixture, parameter
    // placeholders like `{param}` don't substitute, etc.), but we
    // can require that the directory components up through the stage
    // wrapper (`real/fit_<seed>/`) exist on disk. That's the test
    // surface the v1-layout bug actually breaks.
    let mut checked_prefixes: std::collections::BTreeSet<String> =
        std::collections::BTreeSet::new();
    for rel in &documented_paths {
        // Take everything up through the `real/fit_<seed>/` wrapper.
        // Synthetic paths use `synthetic/ds_<NN>/fit_<seed>/`; we
        // skip those because the fixture is real-data only.
        let comps: Vec<&str> = rel.split('/').collect();
        let prefix = if comps.starts_with(&["real", "fit_1"]) || comps.starts_with(&["real"]) {
            "real/fit_1".to_string()
        } else {
            // Unsupported (synthetic, top-level, etc.) — skip.
            continue;
        };
        if checked_prefixes.insert(prefix.clone()) {
            let abs = fit_dir.join(&prefix);
            assert!(
                abs.is_dir(),
                "spec documents paths under `<fit_dir>/{}` but {} does not exist; \
                 the runner did not produce the v2 layout the spec describes",
                prefix,
                abs.display()
            );
        }
    }

    // Belt-and-braces: the runner *must* produce the prefix the spec
    // documents, even if the parser is conservative about which
    // exact paths it asserts. Catches the case where the parser
    // returned only synthetic-only paths and `checked_prefixes`
    // ended up empty.
    assert!(
        !checked_prefixes.is_empty(),
        "no real-data layout prefixes derived from the spec — spec may have been \
         flipped to synthetic-only diagrams without the runner being updated"
    );
}

// ── Deliverable C — `summary ⊆ table` byte-equality ────────────────

/// Force `fit summary --format json`'s `table_row` block to be
/// byte-equal to a `fit table --hash <h> --format json` row for the
/// same fit. Lands live (no `#[ignore]`) per proposal §3 + Tests/CI
/// commitments.
///
/// Map fields (`params`, `ess_posterior`, etc.) on `TableRow` use
/// `BTreeMap` end-to-end so `serde_json` produces lex-ordered keys.
/// If a `HashMap` ever lands in the serialization graph this test
/// will go flaky — that flakiness is the alarm.
#[test]
fn summary_table_row_equals_table_first_row() {
    let camdl = camdl_bin();
    let Some(camdlc) = camdlc_bin() else { return };

    let tmp = tempdir("xpt_c");
    let (ir, data) = build_fixture(&camdlc, tmp.path());
    let output_dir = tmp.path().join("out");
    let fit_toml = write_fit_toml(tmp.path(), &ir, &data, &output_dir);

    let fit_dir = exec_fit_run_v2(&camdl, &fit_toml, &output_dir);

    // The fit-level hash picks the prefix `--hash` filter for `fit table`.
    // Eight characters is the proposal's documented short form. gh#147
    // (M3.2): a CAS fit has no fit-wide `run.json`; the fit-level hash is the
    // `fit` level shared by its stage leaves.
    let full_hash = fit_level_hash(&fit_dir);
    let hash_prefix: String = full_hash.chars().take(8).collect();

    let summary_json = exec_fit_summary_json(&camdl, &fit_dir);
    let summary_row = summary_json
        .get("table_row")
        .cloned()
        .unwrap_or_else(|| panic!("summary JSON missing `table_row`: {}", summary_json));

    // Run `fit table` against the parent of fit_dir (results/fits/),
    // filtered to a single row by --hash.
    let fits_root = output_dir.join("fits");
    let output = std::process::Command::new(&camdl)
        .arg("fit")
        .arg("table")
        .arg(&fits_root)
        .arg("--hash")
        .arg(&hash_prefix)
        .arg("--format")
        .arg("json")
        .output()
        .expect("camdl fit table must invoke");
    assert!(
        output.status.success(),
        "camdl fit table failed: stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let table_doc: serde_json::Value = serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|e| {
            panic!(
                "camdl fit table --format json did not emit valid JSON: {}\nstdout={}",
                e,
                String::from_utf8_lossy(&output.stdout),
            )
        });
    let rows = table_doc
        .get("rows")
        .and_then(|r| r.as_array())
        .unwrap_or_else(|| panic!("table JSON missing `rows` array: {}", table_doc));
    assert_eq!(
        rows.len(),
        1,
        "expected one row matching --hash {}; got {} rows. Doc: {}",
        hash_prefix,
        rows.len(),
        table_doc,
    );
    let table_row = &rows[0];

    // `age_seconds` is the one field that legitimately differs between
    // the two `camdl` invocations: it is `now − created_at` recomputed
    // from the wall clock in each separate process (`build_row` in
    // fit/table_row.rs computes `now_unix - created`, and `now_unix` is
    // `SystemTime::now()` captured independently by `fit summary` and by
    // `fit table`). When the two calls land on opposite sides of a
    // one-second boundary the value differs by 1. `created_at` (the
    // fit's fixed creation time) is identical on both sides, and
    // `age_seconds` is the ONLY wall-clock-derived field in the row —
    // every other field is read from disk-persisted fit artifacts.
    // Normalize it to a canonical value in both rows before comparing
    // (asserting first that the field is present, so its presence and
    // shape are still checked) so the byte-equality assertion covers
    // every other field without racing the clock.
    let normalize_age = |row: &mut serde_json::Value, which: &str| {
        let obj = match row.as_object_mut() {
            Some(o) => o,
            None => panic!("{which} table_row is not a JSON object"),
        };
        assert!(
            obj.contains_key("age_seconds"),
            "{which} table_row is missing the `age_seconds` field",
        );
        obj.insert("age_seconds".into(), serde_json::json!(0));
    };
    let mut summary_row = summary_row;
    let mut table_row = table_row.clone();
    normalize_age(&mut summary_row, "summary");
    normalize_age(&mut table_row, "table");

    // Byte-equality: serialize both to canonical JSON and compare.
    // serde_json::Value::eq is structural (not order-sensitive on
    // objects), but we still want byte-level identity for the
    // `summary ⊆ table` invariant — re-serialize via to_string
    // (which keeps BTreeMap order on Maps) and compare.
    let summary_bytes = serde_json::to_string(&summary_row).unwrap();
    let table_bytes = serde_json::to_string(&table_row).unwrap();
    assert_eq!(
        summary_bytes, table_bytes,
        "summary[\"table_row\"] is not byte-equal to table[\"rows\"][0] \
         (after normalizing the wall-clock `age_seconds` field).\n\
         summary: {}\n\
         table:   {}",
        summary_bytes, table_bytes
    );
}

/// Extract every line from fenced code blocks matching
/// `^\s*<fit_dir>/<rel>` and return `<rel>` (with `<seed>` → `1`
/// substituted, brace-lists expanded, glob/range patterns dropped).
///
/// Fragile-but-loud by design (proposal §B). Does **not** use a
/// markdown AST: walks the file byte-by-byte tracking ` ``` `
/// fences. Lines outside fenced blocks are ignored.
fn parse_layout_diagrams(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut in_fence = false;
    for line in text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if !in_fence {
            continue;
        }
        // Match `<fit_dir>/<rel>` anywhere in the line; allow
        // leading whitespace.
        let stripped = trimmed.strip_prefix("<fit_dir>/");
        let rel = match stripped {
            Some(s) => s,
            None => continue,
        };
        // Trim trailing comment text — TSV/diagram lines often have
        // an inline comment after the path. Heuristic: the path ends
        // at the first whitespace **outside any brace-list**. Brace
        // lists like `{fit_state.toml, mle_params.toml}` contain
        // legitimate internal whitespace and must not be truncated.
        let path_owned: String = {
            let mut depth = 0i32;
            let mut end = rel.len();
            for (i, c) in rel.char_indices() {
                match c {
                    '{' => depth += 1,
                    '}' => depth -= 1,
                    ' ' | '\t' if depth == 0 => { end = i; break; }
                    _ => {}
                }
            }
            rel[..end].to_string()
        };
        let rel = path_owned
            .trim_end_matches(',')
            .trim_end_matches(':')
            .trim_end_matches('.');
        if rel.is_empty() {
            continue;
        }
        // Drop entries with unresolved placeholders / globs.
        // - `<NAME>` — unsubstituted placeholder (we handle <seed>
        //   below; anything else is unsupported).
        // - `{...}` — brace-list or glob. We expand simple comma-
        //   lists below; ranges (`1..N`) and template params
        //   (`{param}`, `{name}`) are dropped.
        let resolved_seeds: Vec<String> =
            substitute_placeholders(rel);
        for r in resolved_seeds {
            for expanded in expand_brace_lists(&r) {
                if expanded.contains('<') || expanded.contains('{') {
                    continue;
                }
                out.push(expanded);
            }
        }
    }
    out
}

/// Replace `<seed>` with `1` (the fixture's default seed). Other
/// `<NAME>` placeholders pass through unchanged so the caller's
/// "drop entries with `<`" filter excludes them.
fn substitute_placeholders(s: &str) -> Vec<String> {
    vec![s.replace("<seed>", "1")]
}

/// Expand `{a, b, c}` into one entry per element. Returns the
/// original string when no brace-list is present, or when the
/// brace-list looks like a range (`{1..N}`) or template
/// (`{param}` / `{name}`).
fn expand_brace_lists(s: &str) -> Vec<String> {
    let lo = match s.find('{') {
        Some(i) => i,
        None => return vec![s.to_string()],
    };
    let hi = match s[lo..].find('}') {
        Some(j) => lo + j,
        None => return vec![s.to_string()],
    };
    let body = &s[lo + 1..hi];
    if body.contains("..") {
        // Range pattern — drop. The caller filters out entries
        // containing `{`, which catches this.
        return vec![s.to_string()];
    }
    if !body.contains(',') {
        // Single-element brace = template placeholder, e.g. `{name}`.
        return vec![s.to_string()];
    }
    let prefix = &s[..lo];
    let suffix = &s[hi + 1..];
    let mut out = Vec::new();
    for elem in body.split(',') {
        let elem = elem.trim();
        // Each elem may itself contain brace-lists — recurse.
        let combined = format!("{}{}{}", prefix, elem, suffix);
        for sub in expand_brace_lists(&combined) {
            out.push(sub);
        }
    }
    out
}

#[cfg(test)]
mod parser_tests {
    use super::*;

    #[test]
    fn extracts_paths_from_simple_fenced_block() {
        let text = "ignored\n\
            ```\n\
            <fit_dir>/real/fit_<seed>/scout/fit_state.toml\n\
            <fit_dir>/real/fit_<seed>/refine/mle_params.toml\n\
            ```\n\
            <fit_dir>/this_should_be_ignored\n";
        let mut paths = parse_layout_diagrams(text);
        paths.sort();
        assert_eq!(
            paths,
            vec![
                "real/fit_1/refine/mle_params.toml".to_string(),
                "real/fit_1/scout/fit_state.toml".to_string(),
            ]
        );
    }

    #[test]
    fn expands_brace_lists() {
        let text = "```\n\
            <fit_dir>/real/fit_<seed>/scout/{fit_state.toml, mle_params.toml}\n\
            ```\n";
        let mut paths = parse_layout_diagrams(text);
        paths.sort();
        assert_eq!(
            paths,
            vec![
                "real/fit_1/scout/fit_state.toml".to_string(),
                "real/fit_1/scout/mle_params.toml".to_string(),
            ]
        );
    }

    #[test]
    fn drops_unresolved_placeholders_and_ranges() {
        let text = "```\n\
            <fit_dir>/real/fit_<seed>/scout/chain_{1..8}/parameter_traces.tsv\n\
            <fit_dir>/real/fit_<seed>/profiles/{param}_profile.tsv\n\
            <fit_dir>/<unknown>/scout/x.toml\n\
            ```\n";
        let paths = parse_layout_diagrams(text);
        assert!(
            paths.is_empty(),
            "range/template/unsubstituted placeholders must drop: {:?}",
            paths
        );
    }

    #[test]
    fn ignores_lines_outside_fenced_blocks() {
        let text =
            "<fit_dir>/real/fit_<seed>/scout/x.toml\n\nThis line has <fit_dir>/foo too.\n";
        let paths = parse_layout_diagrams(text);
        assert!(paths.is_empty());
    }
}

// ── Labels (proposal §5) — end-to-end ──────────────────────────────

/// Full label workflow against the real binary:
///   1. fit run --label "narrow R0, take 1" → label persisted in run.json
///   2. fit table picks up the label, renders it in the row
///   3. fit label <hash> "<new>" rewrites the label
///   4. fit table reflects the new label after the rewrite
#[test]
fn fit_label_workflow_persists_and_surfaces_in_table() {
    let camdl = camdl_bin();
    let Some(camdlc) = camdlc_bin() else { return };

    let tmp = tempdir("xpt_label");
    let (ir, data) = build_fixture(&camdlc, tmp.path());
    let output_dir = tmp.path().join("out");
    let fit_toml = write_fit_toml(tmp.path(), &ir, &data, &output_dir);

    // Step 1: fit run --label "..."
    let initial_label = "narrow R0, take 1";
    let status = std::process::Command::new(&camdl)
        .arg("fit").arg("run")
        .arg("--label").arg(initial_label)
        .arg(&fit_toml)
        .status()
        .expect("camdl fit run must invoke");
    assert!(status.success(), "camdl fit run --label failed");

    let fits = output_dir.join("fits");
    let fit_dir: PathBuf = std::fs::read_dir(&fits).unwrap()
        .flatten().map(|e| e.path()).next()
        .expect("expected one fit dir under fits/");
    let full_hash = fit_level_hash(&fit_dir);
    let hash_prefix: String = full_hash.chars().take(8).collect();

    // Step 2: --label persisted into the fit-level sidecar (gh#147 M3.2). The
    // label is a fit-wide attribute with one authoritative home at the fit
    // segment (`fit.meta.json`) — a CAS fit has no fit-wide run.json, and the
    // label is not redundantly copied onto each stage leaf.
    assert_eq!(sidecar_label(&fit_dir).as_deref(), Some(initial_label),
        "--label must persist into the fit-level sidecar");

    // Step 3: fit table surfaces the label.
    let table_out = std::process::Command::new(&camdl)
        .arg("fit").arg("table").arg(&fits)
        .arg("--hash").arg(&hash_prefix)
        .arg("--format").arg("json")
        .output().expect("camdl fit table must invoke");
    assert!(table_out.status.success(), "camdl fit table failed");
    let table: serde_json::Value = serde_json::from_slice(&table_out.stdout).unwrap();
    let row_label = table["rows"][0]["label"].as_str();
    assert_eq!(row_label, Some(initial_label),
        "table_row.label must reflect Run.label; got {:?}", row_label);

    // Step 4: `camdl label <hash> "<new>"` rewrites.
    let new_label = "narrow R0, take 2";
    let label_status = std::process::Command::new(&camdl)
        .arg("label")
        .arg(&hash_prefix)
        .arg(new_label)
        .arg("--root").arg(&output_dir)
        .status().expect("camdl label must invoke");
    assert!(label_status.success(), "camdl label failed");

    assert_eq!(sidecar_label(&fit_dir).as_deref(), Some(new_label),
        "`camdl label` must rewrite the fit-level sidecar");
    assert_eq!(fit_level_hash(&fit_dir), full_hash,
        "relabel must not change the fit-level hash");

    let table_out2 = std::process::Command::new(&camdl)
        .arg("fit").arg("table").arg(&fits)
        .arg("--hash").arg(&hash_prefix)
        .arg("--format").arg("json")
        .output().unwrap();
    let table2: serde_json::Value = serde_json::from_slice(&table_out2.stdout).unwrap();
    assert_eq!(table2["rows"][0]["label"].as_str(), Some(new_label),
        "table_row.label must reflect the relabel");
}

/// Empty / whitespace-only `--label "<text>"` is rejected at the CLI.
#[test]
fn fit_label_rejects_empty_label_at_cli() {
    let camdl = camdl_bin();
    let Some(camdlc) = camdlc_bin() else { return };
    let tmp = tempdir("xpt_label_empty");
    let (ir, data) = build_fixture(&camdlc, tmp.path());
    let output_dir = tmp.path().join("out");
    let fit_toml = write_fit_toml(tmp.path(), &ir, &data, &output_dir);
    assert!(std::process::Command::new(&camdl)
        .arg("fit").arg("run").arg(&fit_toml)
        .status().unwrap().success());

    let fits = output_dir.join("fits");
    let fit_dir: PathBuf = std::fs::read_dir(&fits).unwrap()
        .flatten().map(|e| e.path()).next().unwrap();
    let hash_prefix: String = fit_level_hash(&fit_dir).chars().take(8).collect();

    for empty in ["", "   "] {
        let out = std::process::Command::new(&camdl)
            .arg("label")
            .arg(&hash_prefix)
            .arg(empty)
            .arg("--root").arg(&output_dir)
            .output().expect("camdl label must invoke");
        assert!(!out.status.success(),
            "empty label {:?} must be rejected", empty);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("empty"),
            "stderr should mention empty: {}", stderr);
    }
}

// ── Issue #22: toml-relative path resolution ──────────────────────

/// `camdl fit run` must resolve relative `[model]` and
/// `[data.observations]` paths against the toml's directory, not
/// the user's CWD. Pre-fix, invoking the binary from a different
/// CWD than the toml's directory broke file lookup; post-fix, the
/// invocation is location-independent (Cargo / pyproject convention).
///
/// Test setup creates a directory tree like:
/// ```
/// <tmp>/proj/
///   fits/he2010.fit.toml      # uses relative paths "../models/..." etc.
///   models/sir.ir.json
///   data/cases.tsv
/// ```
/// then invokes `camdl fit run fits/he2010.fit.toml` from `<tmp>/proj/`
/// (i.e. NOT from inside `fits/`). The fit must succeed — pre-fix it
/// would have failed with "cannot read model at '../models/sir.ir.json'".
#[test]
fn fit_run_resolves_toml_relative_paths_from_any_cwd() {
    let camdl = camdl_bin();
    let Some(camdlc) = camdlc_bin() else { return };

    let tmp = tempdir("xpt_relpath");
    let proj = tmp.path().join("proj");
    let fits_dir = proj.join("fits");
    let models_dir = proj.join("models");
    let data_dir = proj.join("data");
    std::fs::create_dir_all(&fits_dir).unwrap();
    std::fs::create_dir_all(&models_dir).unwrap();
    std::fs::create_dir_all(&data_dir).unwrap();

    // Compile the model into models/, write the data into data/.
    let (ir_in_tmp, data_in_tmp) = build_fixture(&camdlc, tmp.path());
    let ir_target = models_dir.join("sir.ir.json");
    let data_target = data_dir.join("cases.tsv");
    std::fs::rename(&ir_in_tmp, &ir_target).unwrap();
    std::fs::rename(&data_in_tmp, &data_target).unwrap();

    // Write a fit.toml in proj/fits/ that references the model and
    // data via toml-relative paths (../models/..., ../data/...).
    let output_dir = proj.join("results");
    let fit_toml = fits_dir.join("he2010.fit.toml");
    let body = format!(
        r#"
output_dir = "{out}"

[model]
camdl = "../models/sir.ir.json"

[data.observations]
cases = "../data/cases.tsv"

[estimate]
beta  = {{ bounds = [0.01, 5.0], start = 1.0 }}
gamma = {{ bounds = [0.01, 1.0], start = 0.3 }}

[fixed]
N0 = 1000

[stages.scout]
algorithm     = "if2"
backend     = "chain_binomial"
chains     = 2
particles  = 50
iterations = 5
cooling    = 0.7
"#,
        out = output_dir.display(),
    );
    std::fs::write(&fit_toml, body).unwrap();

    // Invoke from `proj/` — NOT from `proj/fits/`. Pre-fix this would
    // have failed because `../models/sir.ir.json` resolved against
    // `proj/` (CWD) and pointed to `models/sir.ir.json` in `proj`'s
    // *parent*. Post-fix, it resolves against `proj/fits/` (the toml's
    // dir) and correctly lands in `proj/models/sir.ir.json`.
    let out = std::process::Command::new(&camdl)
        .current_dir(&proj)
        .arg("fit").arg("run")
        .arg("fits/he2010.fit.toml")
        .output()
        .expect("camdl fit run must invoke");
    assert!(
        out.status.success(),
        "camdl fit run failed when invoked outside the toml's directory.\n  \
         stderr: {}\n  \
         stdout: {}",
        String::from_utf8_lossy(&out.stderr),
        String::from_utf8_lossy(&out.stdout)
    );
    // Sanity: a fit_dir was produced under the configured output_dir.
    let fits_root = output_dir.join("fits");
    let entries: Vec<_> = std::fs::read_dir(&fits_root).unwrap()
        .flatten().map(|e| e.path()).collect();
    assert_eq!(entries.len(), 1,
        "expected one fit dir under {}; got {:?}", fits_root.display(), entries);
}
