//! Integration tests for the experiment-management surface over the fit run
//! store: `fit summary` and `fit table` on what `fit run` stored, labels, and
//! where a fit file's relative paths resolve.
//!
//! - **`fit summary` reads a real run.** Runs `fit run`, then `camdl fit
//!   summary --format json` on the stored run, and asserts the output
//!   contains a non-empty `methods` array.
//!
//! - **`summary ⊆ table` byte-equality.** Asserts that
//!   `summary_json["table_rows"][0]` is byte-equal to
//!   `table_json["rows"][0]` for the same fit. The two surfaces
//!   share one schema; any field added to one without the other
//!   makes this fail.
//!
//! The tests shell out to the built `camdl` and `camdlc.exe` binaries;
//! skipped silently when camdlc is absent so the suite stays runnable in
//! rust-only CI and when tests run before a build. Each fit file names a
//! store of its own in a temporary directory (`store = "..."`).

mod fit_store_support;

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
    projected  = I
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
    let output = Command::new(camdlc).env_remove("CLICOLOR_FORCE").arg(&model_path).output().unwrap();
    assert!(
        output.status.success(),
        "camdlc failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::write(&ir_path, &output.stdout).unwrap();

    // Tiny synthetic data — 10 weekly cases. Doesn't matter for
    // structural tests; we just need the runner to write a method tree.
    let data_path = dir.join("cases.tsv");
    std::fs::write(
        &data_path,
        "time\tcases\n1\t5\n2\t7\n3\t12\n4\t18\n5\t25\n6\t30\n7\t28\n8\t22\n9\t15\n10\t10\n",
    )
    .unwrap();
    (ir_path, data_path)
}

/// Write a tiny IF2 fit.toml, storing into the store `store`, that runs in
/// seconds. 2 chains, 5 iters, 50 particles — enough to store a run
/// without converging on anything meaningful (the tests are structural,
/// not statistical).
fn write_fit_toml(dir: &Path, ir: &Path, data: &Path, store: &Path) -> PathBuf {
    let fit_toml = dir.join("fit.toml");
    let body = format!(
        r#"
store = "{out}"

[model]
camdl = "{ir}"

[data.observations]
cases = "{data}"

[estimate]
beta  = {{ bounds = [0.01, 5.0], start = 1.0 }}
gamma = {{ bounds = [0.01, 1.0], start = 0.3 }}

[fixed]
N0 = 1000

[method]
algorithm     = "if2"
backend     = "chain_binomial"
chains     = 2
particles  = 50
iterations = 5
cooling    = 0.7
"#,
        out = store.display(),
        ir = ir.display(),
        data = data.display(),
    );
    std::fs::write(&fit_toml, body).unwrap();
    fit_toml
}

/// Run `camdl fit run <fit_toml>` and return the one fit run it stored in
/// `store`.
fn exec_fit_run(camdl: &Path, fit_toml: &Path, store: &Path) -> PathBuf {
    let status = Command::new(camdl).env_remove("CLICOLOR_FORCE")
        .arg("fit")
        .arg("run")
        .arg(fit_toml)
        .status()
        .expect("camdl fit run must invoke");
    assert!(status.success(), "camdl fit run failed");
    fit_store_support::the_run(store)
}

/// `camdl fit summary --format json` on the run `fit_dir` of the store `store`.
fn exec_fit_summary_json(camdl: &Path, store: &Path, fit_dir: &Path) -> serde_json::Value {
    let output = Command::new(camdl).env_remove("CLICOLOR_FORCE")
        .env("CAMDL_STORE", store)
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

/// The id of the fit problem a run belongs to: the row's `fit_hash` in
/// `fit table`, which `--hash` filters on.
fn fit_level_hash(run: &Path) -> String {
    fit_store_support::run_inputs(run)["problem_id"].as_str().unwrap().to_string()
}

/// The label `fit table` shows for the one run under `store`.
fn table_label(camdl: &Path, store: &Path) -> Option<String> {
    let out = Command::new(camdl).env_remove("CLICOLOR_FORCE")
        .arg("fit").arg("table").arg(store)
        .arg("--format").arg("json")
        .output().expect("camdl fit table must invoke");
    assert!(out.status.success(), "camdl fit table failed: {}", String::from_utf8_lossy(&out.stderr));
    let table: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let rows = table["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 1, "one run, one row: {table}");
    rows[0]["label"].as_str().map(String::from)
}

// ── `fit summary` reads a real run ─────────────────────────────────

/// `fit summary` on the run `fit run` stored reports its method: a summary
/// that cannot find the method's results reads as "(no MLE methods found)".
#[test]
fn fit_summary_reads_a_stored_fit_run() {
    let camdl = camdl_bin();
    let Some(camdlc) = camdlc_bin() else { return };

    let tmp = tempdir("xpt_a");
    let (ir, data) = build_fixture(&camdlc, tmp.path());
    let store = tmp.path().join("out");
    let fit_toml = write_fit_toml(tmp.path(), &ir, &data, &store);

    let fit_dir = exec_fit_run(&camdl, &fit_toml, &store);

    let json = exec_fit_summary_json(&camdl, &store, &fit_dir);
    let methods = json
        .get("methods")
        .and_then(|s| s.as_array())
        .unwrap_or_else(|| panic!("summary JSON missing `methods` array: {}", json));
    assert!(
        !methods.is_empty(),
        "summary JSON `methods` is empty — the summary found no method result in {}",
        fit_dir.display()
    );
    // gh#901: the old spelling is gone, not kept beside the new one.
    assert!(
        json.get("stages").is_none(),
        "summary JSON must not still carry a `stages` array: {}",
        json
    );

    // The run is stored where the store puts a fit run: an `if2` run folder
    // whose `identity.json` names its kind, with its record beside it.
    let if2_runs = fit_store_support::runs_of_method(&store, "if2");
    assert_eq!(if2_runs, std::slice::from_ref(&fit_dir), "the if2 run is the one run stored");
    assert!(
        fit_dir.join("run.json").is_file(),
        "fit run {} must hold a run.json",
        fit_dir.display()
    );
}

// ── `summary ⊆ table` byte-equality ────────────────────────────────

/// Force `fit summary --format json`'s `table_rows` entry to be
/// byte-equal to a `fit table --hash <h> --format json` row for the
/// same fit.
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
    let store = tmp.path().join("out");
    let fit_toml = write_fit_toml(tmp.path(), &ir, &data, &store);

    let fit_dir = exec_fit_run(&camdl, &fit_toml, &store);

    // The problem id picks the prefix `--hash` filter for `fit table`.
    // Eight characters is the documented short form.
    let full_hash = fit_level_hash(&fit_dir);
    let hash_prefix: String = full_hash.chars().take(8).collect();

    let summary_json = exec_fit_summary_json(&camdl, &store, &fit_dir);
    let summary_rows = summary_json["table_rows"].as_array()
        .unwrap_or_else(|| panic!("summary JSON missing `table_rows`: {}", summary_json));
    assert_eq!(summary_rows.len(), 1, "one completed run, one row: {summary_json}");
    let summary_row = summary_rows[0].clone();

    // Run `fit table` on the store, filtered to a single row by --hash.
    let output = std::process::Command::new(&camdl).env_remove("CLICOLOR_FORCE")
        .arg("fit")
        .arg("table")
        .arg(&store)
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
        "summary[\"table_rows\"][0] is not byte-equal to table[\"rows\"][0] \
         (after normalizing the wall-clock `age_seconds` field).\n\
         summary: {}\n\
         table:   {}",
        summary_bytes, table_bytes
    );
}

/// gh#590: a `pfilter` run beside another method's run of the same problem
/// must not remove the fit from `fit table`, nor leave `fit summary` without
/// the IF2 run's row.
///
/// `pfilter` has no `MethodResult` shape. A row built from a `pfilter` run
/// would fail to load it (`UnknownMethod`) and send the whole fit to the
/// error list — absent from the survey view, reading as "never ran". The row
/// must instead be built from the method that has a result.
#[test]
fn a_pfilter_run_does_not_remove_the_fit_from_fit_table() {
    let camdl = camdl_bin();
    let Some(camdlc) = camdlc_bin() else { return };

    let tmp = tempdir("xpt_pfilter");
    let (ir, data) = build_fixture(&camdlc, tmp.path());
    let store = tmp.path().join("out");

    // Same problem, two methods → one fit problem with an if2 run and a
    // pfilter run (the problem's id hashes the problem, not the method).
    let if2_toml = write_fit_toml(tmp.path(), &ir, &data, &store);
    let if2_body = std::fs::read_to_string(&if2_toml).unwrap();
    let method_at = if2_body.find("[method]").unwrap();
    let pf_dir = tmp.path().join("pf");
    std::fs::create_dir_all(&pf_dir).unwrap();
    let pfilter_toml = pf_dir.join("fit.toml");
    std::fs::write(
        &pfilter_toml,
        format!(
            "{}[method]\nalgorithm = \"pfilter\"\nbackend   = \"chain_binomial\"\n\
             particles = 50\nreplicates = 2\n",
            &if2_body[..method_at]
        ),
    )
    .unwrap();

    let fit_dir = exec_fit_run(&camdl, &if2_toml, &store);
    let status = Command::new(&camdl).env_remove("CLICOLOR_FORCE")
        .arg("fit").arg("run").arg(&pfilter_toml)
        .status()
        .expect("camdl fit run must invoke");
    assert!(status.success(), "camdl fit run (pfilter) failed");

    // Precondition: both runs really are of the one problem, or the test
    // would pass without exercising the mixed problem.
    let problems = fit_store_support::problems(&store);
    assert_eq!(problems.len(), 1, "both methods must share one fit problem: {problems:?}");
    assert_eq!(fit_store_support::runs_of_method(&store, "if2").len(), 1, "no if2 run in {}", store.display());
    assert_eq!(fit_store_support::runs_of_method(&store, "pfilter").len(), 1, "no pfilter run in {}", store.display());

    let output = Command::new(&camdl).env_remove("CLICOLOR_FORCE")
        .arg("fit").arg("table").arg(&store)
        .arg("--format").arg("json")
        .output()
        .expect("camdl fit table must invoke");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "camdl fit table failed: stderr={stderr}");
    let table: serde_json::Value = serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|e| panic!("fit table JSON: {e}\nstdout={}", String::from_utf8_lossy(&output.stdout)));
    let rows = table["rows"].as_array().expect("rows array");
    assert_eq!(
        rows.len(),
        1,
        "the fit with a pfilter run must appear in `fit table`; got {} rows. \
         doc={table} stderr={stderr}",
        rows.len()
    );
    assert_eq!(rows[0]["method"].as_str(), Some("if2"),
        "the row is built from the method that has a result: {}", rows[0]);

    let summary = exec_fit_summary_json(&camdl, &store, &fit_dir);
    let rows = summary["table_rows"].as_array().expect("table_rows");
    assert_eq!(rows.len(), 1, "the IF2 run has a row and the pfilter run has none: {summary}");
    assert_eq!(rows[0]["method"].as_str(), Some("if2"), "{summary}");
}

/// gh#587: a swept fit records each sweep point's own `[fixed]` values in
/// its run's `fit.meta.json`, not the base config's.
///
/// A sweep over a `[fixed]` parameter stores one fit problem per sweep point,
/// whose identity differs by the swept value. A sidecar built once from the
/// base config would record the base value in every run — which
/// `simulate --draws … --fit <run>` then uses to backfill the fixed
/// parameters (`resolve_fixed_for_backfill`).
#[test]
fn a_swept_fit_records_each_sweep_points_own_fixed_values() {
    let camdl = camdl_bin();
    let Some(camdlc) = camdlc_bin() else { return };

    let tmp = tempdir("xpt_sweep_fixed");
    let (ir, data) = build_fixture(&camdlc, tmp.path());
    let store = tmp.path().join("out");
    let fit_toml = write_fit_toml(tmp.path(), &ir, &data, &store); // N0 = 1000

    let status = Command::new(&camdl).env_remove("CLICOLOR_FORCE")
        .arg("fit").arg("run").arg(&fit_toml)
        .arg("--sweep").arg("N0=500,2000")
        .status()
        .expect("camdl fit run must invoke");
    assert!(status.success(), "camdl fit run --sweep failed");

    let mut recorded: Vec<f64> = fit_store_support::runs(&store)
        .into_iter()
        .map(|run| {
            let side = fit_store_support::json(&run.join("fit.meta.json"));
            side["fixed"]["N0"]
                .as_f64()
                .unwrap_or_else(|| panic!("no fixed.N0 in {}: {side}", run.display()))
        })
        .collect();
    recorded.sort_by(f64::total_cmp);
    assert_eq!(
        recorded,
        vec![500.0, 2000.0],
        "each sweep point's fit.meta.json must record the N0 that point \
         held fixed (the base config says 1000)"
    );
}

// ── Labels ─────────────────────────────────────────────────────────

/// Full label workflow against the real binary:
///   1. `fit run --label "narrow R0, take 1"` names the run it stores
///   2. `fit table` shows the label in the run's row
///   3. `camdl label` refuses a fit run: a stored run is sealed, and the
///      refusal leaves its label as it was
///   4. `fit run --label "narrow R0, take 2"` again reuses the stored run and
///      names it by the new label, without changing the problem's id
#[test]
fn fit_label_workflow_persists_and_surfaces_in_table() {
    let camdl = camdl_bin();
    let Some(camdlc) = camdlc_bin() else { return };

    let tmp = tempdir("xpt_label");
    let (ir, data) = build_fixture(&camdlc, tmp.path());
    let store = tmp.path().join("out");
    let fit_toml = write_fit_toml(tmp.path(), &ir, &data, &store);
    let fit_run = |label: &str| {
        let out = Command::new(&camdl).env_remove("CLICOLOR_FORCE")
            .arg("fit").arg("run")
            .arg("--label").arg(label)
            .arg(&fit_toml)
            .output()
            .expect("camdl fit run must invoke");
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        assert!(out.status.success(), "camdl fit run --label failed: {stderr}");
        stderr
    };

    // Step 1: fit run --label "..."
    let initial_label = "narrow R0, take 1";
    fit_run(initial_label);
    let fit_dir = fit_store_support::the_run(&store);
    let full_hash = fit_level_hash(&fit_dir);

    // Step 2: fit table surfaces the label.
    assert_eq!(table_label(&camdl, &store).as_deref(), Some(initial_label),
        "table_row.label must be the run's label");

    // Step 3: `camdl label` refuses the run, and writes nothing.
    let run_id: String = fit_store_support::run_id(&fit_dir).chars().take(12).collect();
    let relabel = Command::new(&camdl).env_remove("CLICOLOR_FORCE")
        .arg("label").arg(&run_id).arg("narrow R0, take 2")
        .env("CAMDL_STORE", &store)
        .output().expect("camdl label must invoke");
    let stderr = String::from_utf8_lossy(&relabel.stderr);
    assert!(!relabel.status.success(), "`camdl label <fit run>` must be refused: {stderr}");
    assert!(stderr.contains("is sealed") && stderr.contains("--label"),
        "the refusal says the run is sealed and how to name it: {stderr}");
    assert_eq!(table_label(&camdl, &store).as_deref(), Some(initial_label), "the refusal wrote nothing");

    // Step 4: the same fit run under another label reuses the stored run.
    let next_label = "narrow R0, take 2";
    let stderr = fit_run(next_label);
    assert!(stderr.contains("cache hit"), "the run is reused, not recomputed: {stderr}");
    assert_eq!(fit_store_support::the_run(&store), fit_dir, "one run is stored");
    assert_eq!(table_label(&camdl, &store).as_deref(), Some(next_label), "table_row.label is the run's label");
    assert_eq!(fit_level_hash(&fit_dir), full_hash, "a label must not change the problem's id");
}

/// An empty or whitespace-only `--label "<text>"` is rejected at the CLI,
/// before anything is stored.
#[test]
fn fit_label_rejects_empty_label_at_cli() {
    let camdl = camdl_bin();
    let Some(camdlc) = camdlc_bin() else { return };
    let tmp = tempdir("xpt_label_empty");
    let (ir, data) = build_fixture(&camdlc, tmp.path());
    let store = tmp.path().join("out");
    let fit_toml = write_fit_toml(tmp.path(), &ir, &data, &store);

    for empty in ["", "   "] {
        let out = Command::new(&camdl).env_remove("CLICOLOR_FORCE")
            .arg("fit").arg("run").arg(&fit_toml)
            .arg("--label").arg(empty)
            .output().expect("camdl fit run must invoke");
        assert!(!out.status.success(),
            "empty label {:?} must be rejected", empty);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("empty"),
            "stderr should mention empty: {}", stderr);
    }
    assert!(fit_store_support::runs(&store).is_empty(), "a refused label stores no run");
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
    let store = proj.join("results");
    let fit_toml = fits_dir.join("he2010.fit.toml");
    let body = format!(
        r#"
store = "{out}"

[model]
camdl = "../models/sir.ir.json"

[data.observations]
cases = "../data/cases.tsv"

[estimate]
beta  = {{ bounds = [0.01, 5.0], start = 1.0 }}
gamma = {{ bounds = [0.01, 1.0], start = 0.3 }}

[fixed]
N0 = 1000

[method]
algorithm     = "if2"
backend     = "chain_binomial"
chains     = 2
particles  = 50
iterations = 5
cooling    = 0.7
"#,
        out = store.display(),
    );
    std::fs::write(&fit_toml, body).unwrap();

    // Invoke from `proj/` — NOT from `proj/fits/`. Pre-fix this would
    // have failed because `../models/sir.ir.json` resolved against
    // `proj/` (CWD) and pointed to `models/sir.ir.json` in `proj`'s
    // *parent*. Post-fix, it resolves against `proj/fits/` (the toml's
    // dir) and correctly lands in `proj/models/sir.ir.json`.
    let out = std::process::Command::new(&camdl).env_remove("CLICOLOR_FORCE")
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
    // Sanity: a fit run was stored in the configured store.
    let runs = fit_store_support::runs(&store);
    assert_eq!(runs.len(), 1,
        "expected one fit run under {}; got {:?}", store.display(), runs);
}
