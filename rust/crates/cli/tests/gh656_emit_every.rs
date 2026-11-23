//! gh#656: `--emit-every`, a per-stream override of a model's emission cadence.
//!
//! `emit_schedule` decides at what times a FORWARD simulation emits synthetic
//! observations. It never enters a likelihood — a fit against real data scores
//! at its data file's own times — so one model can serve a daily and a weekly
//! emission without a regex rewrite of its source.
//!
//! The load-bearing half of this file is IDENTITY. The override is applied at
//! the consumption sites, never by rematerializing the compiled IR the way
//! `--output-every` does, so:
//!
//!   - two cadences address two DISTINCT obs artifacts (they carry different
//!     times, so sharing one address would serve one for the other);
//!   - the same cadence twice addresses the same one;
//!   - the trajectory leaf's `run_id` — the model identity a `fit` against REAL
//!     data shares — does NOT move, so a cosmetic emission change cannot orphan
//!     a completed fit (the gh#653 harm, which an IR rewrite would reintroduce).

mod fit_store_support;

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

/// Two streams on a daily cadence, so a per-stream override has a sibling to
/// leave alone.
const MODEL: &str = r#"
time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate in [0.001, 2.0]
  gamma : rate in [0.001, 1.0]
}

transitions {
  infection : S --> I @ beta * S * (I / (S + I + R))
  recovery  : I --> R @ gamma * I
}

init { S = 990  I = 10  R = 0 }

observations {
  cases {
    columns       { time : time, cases : count }
    covers        = closing_at(time, 1 'days)
    projected     = incidence(infection)
    emit_schedule = every 1 'days
    cases         ~ poisson(rate = projected)
  }
  prevalent {
    columns       { time : time, prevalent : count }
    projected     = I
    emit_schedule = every 1 'days
    prevalent     ~ poisson(rate = projected)
  }
}

simulate { from = 0 'days  to = 28 'days }
"#;

/// One stream whose emission times are a fixed list, not a cadence.
const AT_LIST_MODEL: &str = r#"
time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate in [0.001, 2.0]
  gamma : rate in [0.001, 1.0]
}

transitions {
  infection : S --> I @ beta * S * (I / (S + I + R))
  recovery  : I --> R @ gamma * I
}

init { S = 990  I = 10  R = 0 }

observations {
  cases {
    columns       { time : time, cases : count }
    projected     = I
    emit_schedule = at [0 'days, 7 'days, 14 'days, 21 'days]
    cases         ~ poisson(rate = projected)
  }
}

simulate { from = 0 'days  to = 28 'days }
"#;

fn setup(dir: &Path, src: &str) -> (PathBuf, PathBuf) {
    let m = dir.join("m.camdl");
    std::fs::write(&m, src).unwrap();
    let p = dir.join("p.toml");
    std::fs::write(&p, "beta = 0.5\ngamma = 0.1\n").unwrap();
    (m, p)
}

fn run(args: &[&str]) -> std::process::Output {
    Command::new(binary()).env_remove("CLICOLOR_FORCE")
        .args(args)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

fn skip_if_missing_binary() -> bool {
    if binary().exists() {
        return false;
    }
    eprintln!("skipping: release binary not built; run `make build` first");
    true
}

/// The `time` column of a one-stream obs TSV, found by its header name: a
/// `simulate --obs-dir` table leads with its coordinate columns, a fit's
/// generated dataset does not.
fn obs_times(path: &Path) -> Vec<f64> {
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty());
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let ti = header.iter().position(|h| *h == "time").expect("a `time` column");
    lines.map(|l| l.split('\t').nth(ti).unwrap().parse::<f64>().unwrap()).collect()
}

/// Every simulation cell folder under a run-store root, at any depth: the
/// folders whose `identity.json` records `kind = "sim"`.
fn run_leaves(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if identity_of(&dir).is_some_and(|v| v["kind"] == "sim") {
            out.push(dir.clone());
        }
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                }
            }
        }
    }
    out
}

fn identity_of(dir: &Path) -> Option<serde_json::Value> {
    let txt = std::fs::read_to_string(dir.join("identity.json")).ok()?;
    serde_json::from_str(&txt).ok()
}

/// The `obs/<streams>-<hash>` by-product folder names under a sim cell — the
/// obs artifacts' content addresses.
fn obs_subtree_names(leaf: &Path) -> Vec<String> {
    let mut out: Vec<String> = std::fs::read_dir(leaf.join("obs"))
        .map(|d| {
            d.flatten()
                .filter(|e| e.path().is_dir())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    out.sort();
    out
}

fn run_id_of(leaf: &Path) -> String {
    let v = identity_of(leaf).expect("a sim cell has an identity.json");
    v["run_id"].as_str().expect("identity.json carries a run_id").to_string()
}

// ── Behaviour ────────────────────────────────────────────────────────────────

#[test]
fn the_bare_form_sets_every_streams_cadence() {
    if skip_if_missing_binary() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let (model, params) = setup(tmp.path(), MODEL);
    let dir = tmp.path().join("obs");

    let out = run(&[
        "simulate", model.to_str().unwrap(),
        "--params", params.to_str().unwrap(),
        "--seed", "3",
        "--emit-every", "7",
        "--obs-dir", dir.to_str().unwrap(),
        "--store", tmp.path().join("r").to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "--emit-every 7 must run:\nstderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    // The prevalence stream reads an instant at every emit time, the origin
    // included. The incidence stream's row at the origin would cover the
    // week BEFORE the run and is not written (gh#833): `--emit-every 7`
    // re-widens its declared one-day window to the week it now spans.
    assert_eq!(
        obs_times(&dir.join("prevalent.tsv")),
        vec![0.0, 7.0, 14.0, 21.0, 28.0],
        "'prevalent' must emit weekly, not on its declared daily cadence"
    );
    assert_eq!(
        obs_times(&dir.join("cases.tsv")),
        vec![7.0, 14.0, 21.0, 28.0],
        "'cases' must emit weekly, not on its declared daily cadence"
    );
}

#[test]
fn the_labelled_form_sets_one_stream_and_leaves_its_sibling_alone() {
    if skip_if_missing_binary() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let (model, params) = setup(tmp.path(), MODEL);
    let dir = tmp.path().join("obs");

    let out = run(&[
        "simulate", model.to_str().unwrap(),
        "--params", params.to_str().unwrap(),
        "--seed", "3",
        "--emit-every", "cases=7",
        "--obs-dir", dir.to_str().unwrap(),
        "--store", tmp.path().join("r").to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "--emit-every cases=7 must run:\nstderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    assert_eq!(
        obs_times(&dir.join("cases.tsv")),
        vec![7.0, 14.0, 21.0, 28.0],
        "the named stream must take the override (its origin row covers the \
         week before the run and is not written, gh#833)"
    );
    assert_eq!(
        obs_times(&dir.join("prevalent.tsv")).len(),
        29,
        "the unnamed sibling must keep its declared daily cadence (t = 0..28)"
    );
}

// ── Refusals ─────────────────────────────────────────────────────────────────

#[test]
fn mixing_the_bare_and_labelled_forms_is_refused() {
    if skip_if_missing_binary() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let (model, params) = setup(tmp.path(), MODEL);
    let out = run(&[
        "simulate", model.to_str().unwrap(),
        "--params", params.to_str().unwrap(),
        "--emit-every", "7",
        "--emit-every", "cases=1",
        "--obs-dir", tmp.path().join("obs").to_str().unwrap(),
        "--store", tmp.path().join("r").to_str().unwrap(),
    ]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "mixed forms must be refused\nstderr={stderr}");
    assert!(stderr.contains("mutually exclusive"), "stderr={stderr}");
}

#[test]
fn an_unknown_stream_label_lists_the_valid_ones() {
    if skip_if_missing_binary() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let (model, params) = setup(tmp.path(), MODEL);
    let out = run(&[
        "simulate", model.to_str().unwrap(),
        "--params", params.to_str().unwrap(),
        "--emit-every", "caes=7",
        "--obs-dir", tmp.path().join("obs").to_str().unwrap(),
        "--store", tmp.path().join("r").to_str().unwrap(),
    ]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "a typo'd label must be refused\nstderr={stderr}");
    assert!(
        stderr.contains("'caes' is not an observation stream")
            && stderr.contains("cases")
            && stderr.contains("prevalent"),
        "the error must list the valid labels, got:\n{stderr}"
    );
}

#[test]
fn an_at_list_stream_is_refused_by_name() {
    if skip_if_missing_binary() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let (model, params) = setup(tmp.path(), AT_LIST_MODEL);
    let obs = tmp.path().join("obs.tsv");
    let out = run(&[
        "simulate", model.to_str().unwrap(),
        "--params", params.to_str().unwrap(),
        "--emit-every", "7",
        "--obs", obs.to_str().unwrap(),
        "--store", tmp.path().join("r").to_str().unwrap(),
    ]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "converting a fixed emission list to a cadence must be refused\nstderr={stderr}"
    );
    assert!(
        stderr.contains("'cases'") && stderr.contains("at [...]"),
        "the refusal must name the stream and its declared form, got:\n{stderr}"
    );
    assert!(!obs.exists(), "nothing may be emitted when the override is refused");
}

#[test]
fn a_run_that_emits_no_observations_refuses_the_flag() {
    if skip_if_missing_binary() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let (model, params) = setup(tmp.path(), MODEL);
    let out = run(&[
        "simulate", model.to_str().unwrap(),
        "--params", params.to_str().unwrap(),
        "--emit-every", "7",
        "--store", tmp.path().join("r").to_str().unwrap(),
    ]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "a flag that could do nothing must refuse, not silently no-op\nstderr={stderr}"
    );
    assert!(stderr.contains("emits none"), "stderr={stderr}");
}

#[test]
fn the_dsl_tick_spelling_is_refused_with_a_plain_number_hint() {
    if skip_if_missing_binary() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let (model, params) = setup(tmp.path(), MODEL);
    let out = run(&[
        "simulate", model.to_str().unwrap(),
        "--params", params.to_str().unwrap(),
        "--emit-every", "1 'weeks",
        "--obs-dir", tmp.path().join("obs").to_str().unwrap(),
        "--store", tmp.path().join("r").to_str().unwrap(),
    ]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "stderr={stderr}");
    assert!(
        stderr.contains("tick") && stderr.contains("plain number"),
        "the tick spelling must hint the plain-number form, got:\n{stderr}"
    );
}

// ── Identity ─────────────────────────────────────────────────────────────────

/// Two cadences are two artifacts, and one cadence twice is one artifact.
///
/// The obs by-product is addressed `obs/<streams>-<hash>` under the sim cell,
/// and its hash folds the override. Drop that fold
/// and the two cadences collide on one directory — the second run silently
/// overwriting the first's weekly file with daily rows under an address that
/// still claims to be the first's.
#[test]
fn distinct_cadences_are_distinct_obs_artifacts_and_the_same_cadence_collides() {
    if skip_if_missing_binary() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let (model, params) = setup(tmp.path(), MODEL);
    let store = tmp.path().join("r");

    let sim = |every: &str, obs_dir: &str| {
        let out = run(&[
            "simulate", model.to_str().unwrap(),
            "--params", params.to_str().unwrap(),
            "--seed", "3",
            "--emit-every", every,
            "--obs-dir", obs_dir,
            "--store", store.to_str().unwrap(),
        ]);
        assert!(
            out.status.success(),
            "--emit-every {every} must run:\nstderr={}",
            String::from_utf8_lossy(&out.stderr)
        );
    };

    sim("7", tmp.path().join("o7").to_str().unwrap());
    let leaves = run_leaves(&store);
    assert_eq!(leaves.len(), 1, "one cell, one sim leaf: {leaves:?}");
    let leaf = leaves[0].clone();
    let after_7 = obs_subtree_names(&leaf);
    assert_eq!(after_7.len(), 1, "the first run writes one obs subtree: {after_7:?}");

    // Same cadence again: the same address, no second artifact.
    sim("7", tmp.path().join("o7b").to_str().unwrap());
    assert_eq!(
        obs_subtree_names(&leaf),
        after_7,
        "the same cadence must address the same obs artifact"
    );

    // A different cadence: a second, distinct address alongside the first.
    sim("14", tmp.path().join("o14").to_str().unwrap());
    let after_14 = obs_subtree_names(&leaf);
    assert_eq!(
        after_14.len(),
        2,
        "a different cadence must be a DISTINCT obs artifact, not an \
         overwrite of the first: {after_14:?}"
    );
    assert!(after_14.contains(&after_7[0]), "the weekly artifact must survive: {after_14:?}");

    // And the two carry what they claim to (the origin row's period precedes
    // the run and is not written, gh#833).
    assert_eq!(obs_times(&tmp.path().join("o7").join("cases.tsv")).len(), 4);
    assert_eq!(obs_times(&tmp.path().join("o14").join("cases.tsv")).len(), 2);
}

/// The model identity does not move.
///
/// A `fit` against REAL data keys on its stored compile of the model, as a
/// simulation cell does. Lowering `--emit-every`
/// into the IR — the way `--output-every` does — would move that digest, so a
/// purely cosmetic change to emitted output would orphan a completed fit. It
/// must not: same leaf, same `run_id`, with and without the flag.
#[test]
fn the_flag_does_not_move_the_model_identity_a_real_data_fit_keys_on() {
    if skip_if_missing_binary() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let (model, params) = setup(tmp.path(), MODEL);
    let store = tmp.path().join("r");
    let model_bytes_before = std::fs::read(&model).unwrap();

    let base = |extra: &[&str], obs_dir: &str| {
        let mut args: Vec<String> = vec![
            "simulate".into(), model.to_string_lossy().into(),
            "--params".into(), params.to_string_lossy().into(),
            "--seed".into(), "3".into(),
            "--obs-dir".into(), obs_dir.into(),
            "--store".into(), store.to_string_lossy().into(),
        ];
        args.extend(extra.iter().map(|s| s.to_string()));
        let refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
        let out = run(&refs);
        assert!(
            out.status.success(),
            "run must succeed:\nstderr={}",
            String::from_utf8_lossy(&out.stderr)
        );
    };

    base(&[], tmp.path().join("plain").to_str().unwrap());
    let plain_leaves = run_leaves(&store);
    assert_eq!(plain_leaves.len(), 1);
    let plain_id = run_id_of(&plain_leaves[0]);

    base(&["--emit-every", "7"], tmp.path().join("weekly").to_str().unwrap());
    let after = run_leaves(&store);
    assert_eq!(
        after.len(),
        1,
        "--emit-every must not fork the trajectory leaf — the trajectory bytes \
         do not depend on the emission cadence: {after:?}"
    );
    assert_eq!(
        run_id_of(&after[0]),
        plain_id,
        "the sim run_id — the model identity a real-data fit shares — must be \
         byte-identical with and without --emit-every"
    );
    assert_eq!(
        std::fs::read(&model).unwrap(),
        model_bytes_before,
        "the model source must not be rewritten"
    );
    // Two cadences, two obs artifacts, one trajectory.
    assert_eq!(obs_subtree_names(&after[0]).len(), 2);
}

// ── `fit run` ────────────────────────────────────────────────────────────────

fn camdlc() -> Option<PathBuf> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    p.exists().then_some(p)
}

/// Compile the two-stream model to IR and write a truth file for `[synthetic]`.
fn fit_fixture(dir: &Path) -> (PathBuf, PathBuf) {
    let camdlc = camdlc().expect("checked by the caller");
    let src = dir.join("m.camdl");
    std::fs::write(&src, MODEL).unwrap();
    let out = Command::new(&camdlc).env_remove("CLICOLOR_FORCE").arg(&src).output().unwrap();
    assert!(out.status.success(), "camdlc: {}", String::from_utf8_lossy(&out.stderr));
    let ir = dir.join("m.ir.json");
    std::fs::write(&ir, &out.stdout).unwrap();
    let truth = dir.join("truth.toml");
    std::fs::write(&truth, "beta = 0.5\ngamma = 0.1\n").unwrap();
    (ir, truth)
}

const FIT_METHODS: &str = r#"
[estimate]
beta = { bounds = [0.01, 2.0], start = 0.5 }

[fixed]
gamma = 0.1

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 50
iterations = 3
cooling = 0.7
"#;

fn run_fit(fit_toml: &Path, extra: &[&str]) -> std::process::Output {
    Command::new(binary()).env_remove("CLICOLOR_FORCE")
        .arg("fit")
        .arg("run")
        .arg(fit_toml)
        .args(extra)
        .arg("--no-progress")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("camdl fit run")
}

/// Every stored fit problem and fit run under the store at `out`.
fn fit_bases(out: &Path) -> Vec<PathBuf> {
    let mut v = fit_store_support::problems(out);
    v.extend(fit_store_support::runs(out));
    v
}

/// On a fit with no `[synthetic]` block the flag can do nothing — a fit against
/// real data scores at its data file's own times. Refuse, naming why, rather
/// than accepting a flag whose effect is nil. And nothing may be created: a
/// completed real-data fit keeps its identity.
#[test]
fn fit_run_without_a_synthetic_block_refuses_the_flag_and_re_keys_nothing() {
    if skip_if_missing_binary() || camdlc().is_none() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let (ir, _truth) = fit_fixture(tmp.path());
    let out_dir = tmp.path().join("out");

    // A real-data fit: two daily streams over the model window. Times start at
    // t = 1 — `cases` is an incidence stream, and t = 0 has no preceding
    // accumulation interval.
    let mut data = String::from("time\tcases\tprevalent\n");
    for t in 1..=28 {
        data.push_str(&format!("{t}\t{}\t{}\n", 5 + t % 4, 10 + t % 7));
    }
    let data_path = tmp.path().join("obs.tsv");
    std::fs::write(&data_path, &data).unwrap();

    let fit_toml = tmp.path().join("fit.toml");
    std::fs::write(
        &fit_toml,
        format!(
            r#"store = "{out}"

[model]
camdl = "{ir}"

[data.observations]
cases = "{data}"
prevalent = "{data}"
{methods}
"#,
            out = out_dir.display(),
            ir = ir.display(),
            data = data_path.display(),
            methods = FIT_METHODS,
        ),
    )
    .unwrap();

    let plain = run_fit(&fit_toml, &[]);
    assert!(
        plain.status.success(),
        "the real-data fit must run:\nstderr={}",
        String::from_utf8_lossy(&plain.stderr)
    );
    let bases_before = fit_bases(&out_dir);
    assert_eq!(bases_before.len(), 2, "one fit problem and one run: {bases_before:?}");

    let flagged = run_fit(&fit_toml, &["--emit-every", "7"]);
    let stderr = String::from_utf8_lossy(&flagged.stderr);
    assert!(
        !flagged.status.success(),
        "--emit-every on a real-data fit must refuse, not silently no-op\nstderr={stderr}"
    );
    assert!(
        stderr.contains("[synthetic]") && stderr.contains("never enters the likelihood"),
        "the refusal must say why the flag cannot apply, got:\n{stderr}"
    );
    assert_eq!(
        fit_bases(&out_dir),
        bases_before,
        "a real-data fit must not be re-keyed — or even touched — by this flag"
    );
}

/// `[synthetic]` is the one fit path the cadence reaches: it changes the data
/// that is then FITTED. So the generated file's times change, and — because the
/// fit hashes each training stream's bytes — the fit re-keys onto a new
/// problem and run.
/// That is correct: the data changed.
#[test]
fn a_synthetic_fit_re_keys_because_the_generated_data_changed() {
    if skip_if_missing_binary() || camdlc().is_none() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let (ir, truth) = fit_fixture(tmp.path());

    let fit_for = |tag: &str, out_dir: &Path| -> PathBuf {
        let p = tmp.path().join(format!("fit_{tag}.toml"));
        std::fs::write(
            &p,
            format!(
                r#"store = "{out}"

[model]
camdl = "{ir}"

[synthetic]
true_params = "{truth}"
sim_seeds = [1]
{methods}
"#,
                out = out_dir.display(),
                ir = ir.display(),
                truth = truth.display(),
                methods = FIT_METHODS,
            ),
        )
        .unwrap();
        p
    };

    // Both cadences into one store, so "did it re-key?" is "are there two
    // runs?" rather than a cross-store comparison.
    let out_dir = tmp.path().join("out");
    let toml = fit_for("a", &out_dir);

    // The override names the INSTANT stream: a state read has no window, so
    // any cadence is one the model reads back. The incidence stream declares
    // `closing_at(time, 1 'days)`, so re-spacing IT would write rows the same
    // model cannot read (see the refusal test below).
    for every in ["prevalent=1", "prevalent=7"] {
        let out = run_fit(&toml, &["--emit-every", every]);
        assert!(
            out.status.success(),
            "fit run --emit-every {every} must run:\nstderr={}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    // The generated datasets: each cadence's stored problem holds the data it
    // generated as its training input, one file per stream.
    let generated: Vec<PathBuf> = fit_store_support::problems(&out_dir)
        .into_iter()
        .map(|p| p.join("inputs/prevalent.tsv"))
        .filter(|p| p.is_file())
        .collect();
    // `prevalent` is the stream the override names, and its own file carries
    // only its own rows: 29 daily readings (t = 0…28 inclusive) under
    // `prevalent=1`, and 5 (0, 7, 14, 21, 28) under `prevalent=7`, with no
    // `NA` padding onto a shared time column.
    let mut row_counts: Vec<usize> =
        generated.iter().map(|p| obs_times(p).len()).collect();
    row_counts.sort();
    assert_eq!(
        row_counts,
        vec![5, 29],
        "each cadence must generate its own dataset at its own times — \
         got {generated:?}"
    );
    assert_eq!(generated.len(), 2, "one dataset per cadence: {generated:?}");
    assert_ne!(
        std::fs::read(&generated[0]).unwrap(),
        std::fs::read(&generated[1]).unwrap(),
        "the two cadences must generate different data"
    );

    // …and the fit followed the data onto a distinct run.
    let cells = fit_store_support::runs(&out_dir);
    assert_eq!(
        cells.len(),
        2,
        "each cadence's fit must key on its own generated data: {cells:?}"
    );
}

/// gh#833: a `[synthetic]` dataset is fitted by the model that generated it,
/// so re-spacing a stream whose declared window has another width would write
/// rows the model reads as gapped. Refused, naming the declaration and the one
/// that would emit the requested cadence — not rescaled behind the user's back.
#[test]
fn a_synthetic_fit_refuses_an_emit_every_that_contradicts_the_declared_window() {
    if skip_if_missing_binary() || camdlc().is_none() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let (ir, truth) = fit_fixture(tmp.path());
    let toml = tmp.path().join("fit.toml");
    std::fs::write(
        &toml,
        format!(
            "store = \"{out}\"\n\n[model]\ncamdl = \"{ir}\"\n\n[synthetic]\n\
             true_params = \"{truth}\"\nsim_seeds = [1]\n{methods}\n",
            out = tmp.path().join("out").display(),
            ir = ir.display(),
            truth = truth.display(),
            methods = FIT_METHODS,
        ),
    )
    .unwrap();

    let out = run_fit(&toml, &["--emit-every", "7"]);
    assert!(!out.status.success(), "a weekly override on a daily-declared incidence stream must be refused");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("cases") && err.contains("covers") && err.contains("closing_at(time, 7 'days)"),
        "the refusal must name the stream, its declaration and the declaration that would emit weekly: {err}");
}
